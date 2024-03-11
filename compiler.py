#compiler.py

import asyncio
import logging
from llm_compiler.agent import WebDevAgent, LLM

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.llms import BaseLLM
from langchain.prompts.base import StringPromptValue
from src.base import Tool

class WebDevLLMCompiler:
    """
    Web Development LLM Compiler Engine that orchestrates the interaction
    with language models and tools for generating and processing content.
    """

    def __init__(self, api_key: str, model_name: str, tools):
        self.api_key = api_key
        self.model_name = model_name
        self.tools = tools
        self.client = OpenAI(api_key=self.api_key)

    async def generate_content(self, input_query: str) -> str:
        """
        Generates content based on a given input query by asynchronously
        invoking the LLM and applying necessary tools for post-processing.

        Args:
            input_query: User input query to guide content generation.

        Returns:
            The generated content as a string.
        """
        # Construct the complete prompt
        complete_prompt = f"{self.prompt}\n{input_query}"

        # Generate initial content using the LLM
        initial_content = await self.llm.predict(complete_prompt, max_tokens=self.max_tokens)

        # Apply post-processing tools if any
        for tool in self.tools:
            if asyncio.iscoroutinefunction(tool.run):
                initial_content = await tool.run(initial_content)
            else:
                initial_content = tool.run(initial_content)

        return initial_content