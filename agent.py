# agent.py

from __future__ import annotations
from openai import OpenAI
from langchain.schema.language_model import BaseLanguageModel
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from .gpt_prompts import PLANNER_PROMPTS, OUTPUT_PROMPTS
from llm_compiler.tools import WebContentTool, DalleTool
from src.base import Tool
from .llm_chain import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.manager import Callbacks
from langchain.agents.agent import AgentOutputParser, BaseSingleActionAgent
from typing import Type, Any, Dict, List, Optional, Tuple, Union
from llm_compiler.compiler import WebDevLLMCompiler
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Class for interaction with OpenAI GPT models
class LLM(BaseLanguageModel):
    model_name: str = Field(default="gpt-3.5-turbo-1106")
    api_key: str
    client: Any # Declare the client field

    class Config:
        arbitrary_types_allowed = True  # Allowing arbitrary types

    def __init__(self, **data):
        super().__init__(**data)
        self.client = OpenAI(api_key=self.api_key)

    def predict(self, user_input: str, max_tokens: int = 500) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content if response.choices else None
        except Exception as e:
            logger.error(f"Error in OpenAI prediction: {e}")
            return None

    # Implement abstract methods with minimal implementations
    def agenerate_prompt(self, *args, **kwargs):
        pass

    def apredict(self, *args, **kwargs):
        pass

    def apredict_messages(self, *args, **kwargs):
        pass

    def generate_prompt(self, *args, **kwargs):
        pass

    def invoke(self, *args, **kwargs):
        pass

    def predict_messages(self, *args, **kwargs):
        pass
class WebPageProcessor:
    def __init__(self, llm: LLM):
        self.llm = llm

    # Extracts visible text from HTML content
    def extract_visible_text(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        return soup.get_text(separator='\n', strip=True)

    # Modifies text using GPT based on user query
    def modify_text_with_gpt(self, text: str, user_query: str) -> str:
        combined_input = f"{user_query}\n{text}"
        modified_text = self.llm.predict(combined_input)
        return modified_text if modified_text else text

# Web Development Agent class
class WebDevAgent:
    def __init__(self, llm: LLM, tools: List[Tool], webpage_processor: WebPageProcessor):
        self.llm = llm
        self.tools = tools
        self.webpage_processor = webpage_processor

    def setup_tools(self):
        self.agent = WebDevAgent(llm=self.llm)
        self.tools = [Tool(name="tool1", agent=self.agent), Tool(name="tool2", agent=self.agent)]

    # Extracts visible text from HTML content
    def extract_visible_text(self, html_content=None):
        return self.webpage_processor.extract_visible_text(html_content)

    def modify_text_with_gpt(self, user_query=None):
        return self.webpage_processor.modify_text_with_gpt(text, user_query)

    # Processes webpage, modifying content based on GPT outputs
    def process_webpage(self, html_content: str, user_query: str) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        visible_text = self.extract_visible_text(str(soup))
        modified_text = self.modify_text_with_gpt(visible_text, user_query)
        main_content = soup.find('div', {'id': 'modified-scroll'})  # Adjust as needed
        if main_content:
            main_content.clear()
            main_content.append(BeautifulSoup(modified_text, 'html.parser'))
        return str(soup)

    # Method to modify web content based on GPT model outputs
    def modify_content_with_gpt(self, html_content: str, user_input: str) -> str:
        modified_text = self.llm.predict(user_input)
        if not modified_text:
            return html_content

        logger.info("Modifying content with GPT.")
        modified_text = self.llm.predict(user_input)
        if not modified_text:
            return html_content

        soup = BeautifulSoup(html_content, 'html.parser')
        target_div = soup.find('div', {'id': 'modified-scroll'})
        if target_div:
            target_div.clear()
            target_div.append(BeautifulSoup(modified_text, 'html.parser'))
            logger.info(f"Modified content: {str(soup)[:500]}")
            return str(soup)

    def modify_textual_content(self, soup: BeautifulSoup, user_input: str) -> BeautifulSoup:
        """
        This method modifies the textual content of a given BeautifulSoup object
        based on user input. It uses the LLM to generate modified text and replaces
        the text in the specified section of the HTML.
        """
        section_to_modify = self.determine_focus_areas(user_input)
        section = soup.find('div', {'id': 'modified-scroll'}) if section_to_modify else soup.body
        if section:
            modified_text = self.llm.predict(user_input)
            if modified_text:
                section.string = modified_text
            else:
                logger.warning("No modification made, using original text.")
                return soup

    def determine_focus_areas(self, user_input: str) -> str:
        """
        Determines the focus area of the webpage to be modified based on user input.
        Returns the ID of the HTML section to be modified.
        """
        focus_areas_keywords = {
        "header": ["header", "top bar", "logo"],
        "footer": ["footer", "bottom section"],
        "main_content": ["content", "articles", "posts"],
        "navigation": ["menu", "navigation bar", "links"],
        "SEO": ["SEO", "search engine optimization", "meta tags"]
        }

        for area, keywords in focus_areas_keywords.items():
            if any(keyword in user_input.lower() for keyword in keywords):
                return area  # Return the ID of the HTML element

        return "main_content"  # Default focus area

    def generate_modifications_with_gpt(self, user_input: str, html_section: str) -> str:
        """
        Generates modifications for a specific section of HTML using GPT models.
        """
        try:
            gpt_response = self.llm.predict(user_input)
            return gpt_response if gpt_response else html_section
        except Exception as e:
            logger.error("Error in generate_modifications_with_gpt: %s", e)
            return html_section
            
    def process_webpage(self, html_content: str, user_query: str) -> str:
                """Processes the entire webpage, modifying text as needed."""
                soup = BeautifulSoup(html_content, 'html.parser')
                visible_text = self.extract_visible_text(str(soup))
                modified_text = self.modify_text_with_gpt(visible_text, user_query)

                # Logic to find and replace main content in HTML
                main_content = soup.find('div', {'id': 'modified-scroll'})  # Adjust based on actual HTML structure
                if main_content:
                    main_content.clear()  # Clear existing content
                    main_content.append(BeautifulSoup(modified_text, 'html.parser'))

                    return str(soup)

    def parse_original_webpage(self, file_path: str) -> BeautifulSoup:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return BeautifulSoup(file.read(), 'html.parser')

    def modify_textual_content(self, soup: BeautifulSoup, user_input: str) -> BeautifulSoup:
                        section_to_modify = self.determine_focus_areas(user_input)
                        section = soup.find('div', {'id': 'modified-scroll'})
                        if section:
                            modified_text = self.llm.predict(user_input)
                            section.string = modified_text
                            return soup

    def determine_section_to_modify(self, user_input):
        # Simple keyword-based logic to determine the section
        if 'header' in user_input.lower():
            return 'header'
        elif 'footer' in user_input.lower():
            return 'footer'
        # Add more conditions for other sections
        else:
            return 'main'  # default section

    def extract_section(self, html_content, section_id, focus_areas_keywords=None, user_input=None):
        """
        Extracts a specific section from the HTML content based on a section identifier.
        :param html_content: String containing the full HTML content.
        :param section_id: String identifier for the section to extract (e.g., 'header', 'footer').
        :return: String containing HTML of the extracted section.
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # Example logic to extract a section based on ID
        if section_id == 'header':
            # Adjust based on actual HTML structure
            section = soup.find('header')
        elif section_id == 'footer':
            # Adjust based on actual HTML structure
            section = soup.find('footer')
        # Add more conditions for other sections
        else:
            section = soup.find('div', {'id': 'modified-scroll'})  # Default section; adjust as needed

            return str(section) if section else ""

        # Initialize focus_areas
        focus_areas = []

        # Determine focus areas based on keywords in user_input
        for area, keywords in focus_areas_keywords.items():
            if any(keyword in user_input.lower() for keyword in keywords):
                focus_areas.append(area)

        # If no specific area is mentioned, consider all areas
        if not focus_areas:
            focus_areas = list(focus_areas_keywords.keys())

            return focus_areas            

    def execute_tool(self, tool_name, tool_args):
                if tool_name == 'web_content_tool':
                    return self.web_content_tool.generate(*tool_args)
                elif tool_name == 'dalle_tool':
                    return self.dalle_tool.generate(*tool_args)
                else:
                    return f"Unknown tool: {tool_name}"

    def generate_output(self, tool_output):
                if isinstance(tool_output, str):
                    return f"Formatted Output: {tool_output}"
                else:
                    return f"Complex Output: {tool_output}"

class WebDevAgentOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # Parsing logic to interpret language model outputs
        if "HTML:" in text:
            html_content = self.extract_content_after_label(text, "HTML:")
            return AgentAction({"html": html_content})
        elif "CSS:" in text:
            css_content = self.extract_content_after_label(text, "CSS:")
            return AgentAction({"css": css_content})
        elif "JavaScript:" in text:
            js_content = self.extract_content_after_label(text, "JavaScript:")
            return AgentAction({"javascript": js_content})
        elif "DALL-E:" in text:
            image_query = self.extract_content_after_label(text, "DALL-E:")
            return AgentAction({"dalle_query": image_query})
        else:
            # Default action or error handling
            return AgentFinish({"output": "Unrecognized response format"}, "")

    def extract_content_after_label(self, text, label):
                """Extracts content after a specific label in the text."""
                try:
                    start_index = text.index(label) + len(label)
                    content = text[start_index:].strip()
            # Extracts the first line after the label
                    return content.split("\n")[0].strip()
                except ValueError:
                    return ""

