#chat_agent.py

import re
import os
import openai
from typing import Any, List, Optional, Sequence, Tuple
from langchain.agents.agent import AgentOutputParser
from langchain.agents.structured_chat.output_parser import StructuredChatOutputParserWithRetries
from langchain.agents.structured_chat.prompt import PREFIX, SUFFIX
from langchain.callbacks.base import BaseCallbackManager
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import AgentAction, BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool
from llm_compiler.llm_chain import LLMChain
from .tools import WebContentTool, DalleTool
from .agent import WebDevAgentOutputParser

HUMAN_MESSAGE_TEMPLATE = "{input}\n\n{agent_scratchpad}"

# Initialize OpenAI
openai.api_key = os.getenv('openai_api_key')

# Set model name
model_name = "gpt-3.5-turbo-1106"

class WebDevChatAgent:
    """Web Development Chat Agent."""

    def __init__(self, llm: BaseLanguageModel, tools: Sequence[BaseTool]):
        self.llm_chain = LLMChain(llm=llm, model_name=model_name, api_key=openai.api_key, prompt=self.create_prompt(tools))
        self.output_parser = WebDevAgentOutputParser()
        self.web_content_tool = WebContentTool()
        self.dalle_tool = DalleTool()

    def process_input(self, user_input: str):
        # Process the input and generate web content
        action, observation = self.llm_chain.predict(user_input)
        return self.output_parser.parse(action)

    @staticmethod
    def create_prompt(tools: Sequence[BaseTool]) -> BasePromptTemplate:
        tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        template = f"{PREFIX}\n{tool_descriptions}\n{SUFFIX}"
        return ChatPromptTemplate(
            input_variables=["input", "agent_scratchpad"],
            messages=[
                SystemMessagePromptTemplate.from_template(template),
                HumanMessagePromptTemplate.from_template(HUMAN_MESSAGE_TEMPLATE)
            ]
        )

    def execute_tool(self, tool_name, tool_args):
        # Execute the appropriate tool from tools.py
        if tool_name == 'web_content':
            return self.web_content_tool.generate(tool_args)
        elif tool_name == 'dalle':
            return self.dalle_tool.generate(tool_args)
        else:
            return "Unknown tool requested."

# Example usage
# llm = ... # Initialize your language model
# tools = [WebContentTool(), DalleTool()]
# chat_agent = WebDevChatAgent(llm, tools)