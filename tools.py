#tools.py

import os
import openai
from langchain.tools import BaseTool
from pydantic import Field
from typing import Any

MODEL_GPT3_TURBO = "gpt-3.5-turbo-1106"
MODEL_DALLE_3 = "dall-e-3"


# Modify BaseTool to accept client as an attribute
class BaseTool(BaseTool):
    client: Any = Field(None)  # Add the client field

    class Config:
        arbitrary_types_allowed = True  # Allowing arbitrary types in Pydantic models


class WebContentTool(BaseTool):
    """Tool for generating web content using GPT-3.5 Turbo 1106."""
    name = "web_content"
    description = "Generates HTML, CSS, or JavaScript based on user input."

    def _run(self, prompt, model=MODEL_GPT3_TURBO):
        """Generate web content."""
        response = self.client.chat.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=500
        )
        return response.choices[0].text.strip()


class DalleTool(BaseTool):
    """Tool for generating images using DALL-E 3."""
    name = "dalle"
    description = "Generates images based on text descriptions."

    def _run(self, prompt):
        """Generate image using DALL-E 3."""
        try:
            response = self.client.Image.generate(
                model=MODEL_DALLE_3,
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            return response.data[0].url
        except Exception as e:
            print(f"DEBUG: Image generation failed with error: {str(e)}")
            return f"Image generation is currently unavailable. Please try again later."


class InvalidTool(BaseTool):
    """Tool run when an invalid tool name is encountered."""
    name = "invalid_tool"
    description = "Called when tool name is invalid. Suggests valid tool names."

    def _run(self, requested_tool_name, available_tool_names):
        available_tool_names_str = ", ".join(available_tool_names)
        return f"{requested_tool_name} is not a valid tool, try one of [{available_tool_names_str}]."


# Create an OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv('openai_api_key'))

tools = [
    WebContentTool(client=openai_client),
    DalleTool(client=openai_client),
    InvalidTool(client=openai_client)
]
