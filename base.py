# base.py

from __future__ import annotations
import asyncio
import inspect
import threading
from functools import partial
from typing import Type, Any, Awaitable, Callable, Optional, Union
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Extra, Field, create_model, validate_arguments
from langchain.schema.runnable import RunnableConfig
from langchain.tools import BaseTool
from llm_compiler.config import openai_api_key

class ToolException(Exception):
    """Exception for tool execution errors."""
    pass

class Type:
    pass
class Tool(BaseTool):
    """A class to represent a generic tool that can execute both synchronous and asynchronous functions."""

    def __init__(self, name: str, func: Union[Callable[..., Any], Callable[..., Awaitable[Any]]],
                 description: str = ""):
        self.name = name
        self.func = func
        self.description = description

    async def async_run(func, *args, **kwargs):
        """
        Executes an asynchronous function, ensuring that an event loop is available.
        This function is designed to be used in environments where the default event loop
        might not be available or is already running.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # If no running loop is found, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            # If the loop is running but not in the current thread, create a new loop for the current context
            if not loop.is_running() or isinstance(threading.current_thread(), threading._MainThread):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run synchronous functions in the default executor to avoid blocking the event loop
            return await loop.run_in_executor(None, func, *args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        """Public method to run the tool's function. Automatically handles synchronous and asynchronous functions."""
        try:
            return asyncio.run(self._run_async(*args, **kwargs))
        except RuntimeError as e:
            # Fallback for running asyncio.run in an already running event loop (e.g., Jupyter notebooks, certain web servers)
            if "There is no current event loop in thread" in str(e):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(self._run_async(*args, **kwargs))
            else:
                raise


class StructuredTool(BaseTool):
    args_schema: Optional[Type[BaseModel]] = None
    """Tool that can operate on multiple inputs."""

    description: str = ""
    args_schema: Type[BaseModel]
    func: Optional[Callable[..., Any]]
    coroutine: Optional[Callable[..., Awaitable[Any]]] = None

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Use the tool."""
        if self.func:
            return self.func(*args, **kwargs)
        raise NotImplementedError("Tool does not support sync")

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Use the tool asynchronously."""
        if self.coroutine:
            return await self.coroutine(*args, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(None, partial(self._run, **kwargs), *args)

def tool(func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> Tool:
    """Decorator to create tools from functions."""
    name = name or func.__name__
    description = description or func.__doc__ or "No description provided."
    return Tool(name=name, func=func, description=description)

# Example usage of the tool decorator
# @tool(name="ExampleTool", description="This is an example tool.")
# def example_tool(input_str: str) -> str:
#     return f"Processed: {input_str}"