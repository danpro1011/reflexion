import re
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import pprint as pp
from util import translate_text

from llm import AnyOpenAILLM
from prompts import debate_meta_reflection_prompt, debator_response_prompt, debator_initial_prompt, consensus_reached_prompt, determine_consensus_prompt
from fewshots import REFLECTIONS

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate
try:
    from langchain.schema import HumanMessage, SystemMessage, AIMessage, ChatMessage
except ImportError:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ChatMessage

try:
    from langchain_openai import ChatOpenAI, OpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
    from langchain.llms import OpenAI

from dotenv import load_dotenv
load_dotenv()


FINISH_PATTERN = re.compile(r"Finish\\[(.*?)\\]", re.IGNORECASE)


def extract_answer(completion: str) -> str:
    """
    Best-effort extraction of the argument passed to Finish[...].
    Falls back to the raw completion if no Finish token is found.
    """
    for line in reversed(completion.splitlines()):
        match = FINISH_PATTERN.search(line)
        if match:
            return match.group(1).strip()
    match = FINISH_PATTERN.search(completion)
    if match:
        return match.group(1).strip()
    return completion.strip()

class DebateLLM:
    def __init__(
        self,
        question: str,
        scratchpad: str,
        debate_id: int,
        llm: Optional[AnyOpenAILLM] = None,
        system_prompt: PromptTemplate =  debate_meta_reflection_prompt,
        language: str = "en", #NOTE: This has to be the langauge string as specified by (https://docs.cloud.google.com/translate/docs/languages#nmt)
        translator: str = "bing"
    ) -> None:
        self.question = question
        self.scratchpad = scratchpad
        self.llm = llm or AnyOpenAILLM(
            temperature=.25,
            max_tokens=256,
            model_name="gpt-3.5-turbo",
            model_kwargs={"stop": "\n"}, #Makes it stop after a new line, to limit long responses
        )
        self.debate_id = debate_id
        self.language = language
        self.translator = translator

        #This stays constant so convienient to just save it here 
        self.system_prompt = SystemMessage(content=system_prompt.format(examples=REFLECTIONS))

    def initial_response(self) -> str:
        #Idea here is simple enough, just translate every prompt given to other language for its 'thinking' process, and then translate back
        prompt = debator_initial_prompt.format(
                question=self.question,
                scratchpad=self.scratchpad
                )
        
        final_messages = [self.system_prompt, HumanMessage(content=prompt)]

        if self.language != "en":
            final_messages = [self.translate_msg(m, self.language) for m in final_messages]

        output = self.llm.query(final_messages)

        if self.language != "en":
            output = translate_text(output, source_language=self.language, target_language="en")

        return output

    # The prompt for this should pretty much always be the same, so I don't see the need to pass it in as an argument
    def debate_response(self, debate_history) -> str:
        question_msg = HumanMessage(content=
        f"Previous Trial:\n"
        f"Question: {self.question}\n"
        f"{self.scratchpad}\n"
        f"These are the reflections that other agents analyzing your reasoning traces came up with:"
        )
        debate_messages = self._format_debate_history(debate_history)

        final_messages = [
            self.system_prompt,
            question_msg,
            *debate_messages,
            HumanMessage(content="Using the opinions of other agents as additional advice, give an updated response...")
        ]

        # translate (if needed)
        if self.language != "en":
            final_messages = [self.translate_msg(m, self.language) for m in final_messages]

        output = self.llm.query(final_messages)

        if self.language != "en":
            output = translate_text(output, source_language=self.language, target_language="en")

        return output
        if self.language != "en":
            # Parts of the prompt that need to be translated
            question_context = f"Previous Trial:\nQuestion{self.question}{self.scratchpad}\nThese are the reflections that other agents analyzing your reasoning traces came up with:"
            response_question = "Using the opinion of other agents as additional advice, can you give an updated response ..." 
            debate_history = self._format_debate_history(debate_history, as_pormpt=False)
            
            # Use translation library to translate the parts
            question_context = translate_text(question_context, source_language = "en", target_language = self.language)
            debate_history = [translate_text(x, source_language = "en", target_language = self.language) for x in debate_history]
            response_question = translate_text(response_question, source_language = "en", target_language = self.language)

            # Format them in langchain 
            question_context = HumanMessage(content = question_context)
            response_question = HumanMessage(content = response_question)
            #TODO: This is obviously not how it should be done, temporary for just initial testing
            debate_history = [HumanMessage(content = x) for x in debate_history]

            
            response = self.llm.query([self.system_prompt, question_context, *debate_history, response_question])
            response = translate_text(response, source_language = self.language, target_language = "en")
            
        else:
            question_context = f"Previous Trial:\nQuestion{self.question}{self.scratchpad}\nThese are the reflections that other agents analyzing your reasoning traces came up with:"
            question_context = HumanMessage(content = question_context)

            debate_history = self._format_debate_history(debate_history)

            response_question = HumanMessage(content = "Using the opinion of other agents as additional advice, can you give an updated response ...")
            #Order is system_prompt + question/scratchpad context + debate_history + finally the question
            response = self.llm.query([self.system_prompt, question_context, *debate_history, response_question])

        return response
    
    def _format_debate_history(self, debate_history) -> List:
        """
        Because of how the old LangChain library worked, this function is needed. It basically takes the debate history
        and formats it so that the LLM knows that the responses that it gave came from itself
        """

        # Capture groups:
        #   (1) debator ID
        #   (2) message content
        pattern = re.compile(
            r"Debator\s+(\d+):\s*(.*?)\n(?=Debator\s+\d+:|$)",
            re.DOTALL
        )

        matches = pattern.findall(debate_history)
        formatted_messages = []

        for debator_id, content in matches:
            debator_id = int(debator_id)

            if debator_id == self.debate_id:
                formatted_messages.append(
                    AIMessage(content=content.strip())
                )
            else:
                # It's recommended to use 'chat message' + role assistant for task like this rather than have it be part of 'human message' 
                formatted_messages.append(
                    ChatMessage(
                        role="assistant",
                        name=f"debator_{debator_id}",
                        content=content.strip()
                    )
                )

        return formatted_messages
    
    def translate_msg(self, msg, target):
        '''Helper function that translates LangChain message while still keeping class structure intact'''
        return type(msg)(content=translate_text(
            msg.content,
            source_language="en",
            target_language=target,
        ))

#TODO: This class is written around the assumption that we're just using the openai llm, but it really should work for any LLM type
class DebateCoordinator:
    def __init__(
        self,
        question: str,
        answer_key: str,
        # scratchpad: str,
        max_num_rounds: int = 5, #This is the hard max, not recommended or average number of runs
        llm: AnyOpenAILLM = None
    ) -> None:
        self.question = question
        self.answer_key = answer_key
        self.max_num_rounds = max(1, max_num_rounds)
        self.round_number = 0
        self.debate_history = ""
        
        #TODO: Better functionality for changing the LLM model
        self.model_name = "gpt-3.5-turbo" 

        self.llm = llm or AnyOpenAILLM(
            temperature=0,
            max_tokens=256,
            model_name=self.model_name,
        )
        #NOTE: Do we want the 'stop' thingy here too or not


    def _build_debators(self, num_debators: int, scratchpad:str, llm= None) -> List[DebateLLM]:
        debators: List[DebateLLM] = []
        # Idea behind this implementation is that altering the language is enough to alter the output, a higher temp shouldn't be needed
        for indx in range(num_debators):
            llm = AnyOpenAILLM(
                temperature=0,
                max_tokens=256,
                model_name="gpt-3.5-turbo",
                model_kwargs={"stop": "\n"},
            )

            #TODO: Better way to specify languages of debators from the Debate Coordinator interface
            if indx == 0:
                language = "en"
            if indx == 1:
                language = "es"
            # if indx == 2:
            #     language = "es"
            
            debators.append(
                DebateLLM(
                    question=self.question,
                    scratchpad=scratchpad,
                    llm=llm,
                    debate_id=indx,
                    language=language
                )
            )
        return debators

    #TODO: Add some type of logging for this
    def run(self, num_debators, scratchpad) -> str:
        #Context has to be passed into this and debators have to be build just for that instance, 
        # just based off of how its designed right now
        rounds: List[List] = []

        debators = self._build_debators(num_debators,scratchpad) 
        
        num_debate_rounds = 0
        curr_round = []
        consensus = ""
        while(num_debate_rounds < self.max_num_rounds):
            #First round you have to generate the initial responses, after that it's all the same
            if num_debate_rounds == 0:
                for debator in debators:
                    response = debator.initial_response()
                    curr_round.append(response)
            
            else:
                for debator in debators:
                    response = debator.debate_response(self.debate_history)
                    curr_round.append(response)

            self._update_debate_history(curr_round)
            num_debate_rounds += 1
            rounds.append(curr_round)
            curr_round = []

            consensus_reached, debator_id = self._find_consensus()

            consensus = self._extract_debator_response(debator_id)
            
            if consensus_reached:
                break

        #If we reached the max amount of rounds without arriving at a consensus, we must extract some type of consensus
        if not consensus_reached:
            print("No consensus reached")
            prompt = HumanMessage(content = determine_consensus_prompt(debate_log = self.debate_history))
            consensus = self.llm(prompt)
            
        print("--"*20 + "Full debate log" + "--"*20)
        pp.pprint(rounds)

        return consensus
    

    def _find_consensus(self) -> tuple[bool, str]:
        """
        Helper function that views the debate log and determines whether or not a verdict has been reached. 
        Right now just having the 'judge' llm do this, but there are other potential methods to accomplish a similar task
        """
        verdict = self.llm(consensus_reached_prompt.format(debate_log=self.debate_history))
        #Apparently, python syntax is different from json syntax, and if the bool is loaded pythonically, json.loads crashes
        safe = verdict.replace("True", "true").replace("False", "false").replace("None", "null")
        verdict = json.loads(safe)
        # print(verdict)

        return verdict["consensus_reached"], verdict.get("debator_id",-1)
    
    def _extract_reflection(self, text) -> str:
        """
        Assuming that the output passed into text is of the format '<sometext> Reflection[<sentence>].
        And we extract the <sentence> inside of the Reflection block.
        """
        match = re.search(r"Reflection\[(.*?)\]", text, re.DOTALL)
        if not match:
            print("Not able to extract the reflection from ", text)
            return ""
        return match.group(1).strip()
        
    def _extract_debator_response(self, target_debator_id) -> str:
        pattern = re.compile(r"Debator (\d+):\s*(.*?)\n(?=Debator \d+:|$)", re.DOTALL)
        matches = pattern.findall(self.debate_history)

        resp = ""
        for debator_id, text in matches:
            if int(debator_id) == target_debator_id:
               resp = text
        
        if resp == "": print("ERROR: debator id not found")
        return resp
    
    def _update_debate_history(self, new_round):
        #Tracking the round may not be necessary
        self.debate_history += "--"*5 + f"Start of round {self.round_number}" + "--"*5 + '\n'

        for indx, response in enumerate(new_round):
            self.debate_history += f"Debator {indx}: " + response + "\n"
            
        self.round_number += 1

    #TODO: Some function that does this is likely gonna be needed
    def _summarize_debate_history(self):
        ...