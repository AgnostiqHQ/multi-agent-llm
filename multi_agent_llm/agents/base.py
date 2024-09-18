import asyncio
from concurrent.futures import Future
from html import escape
from typing import Any, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field

from ..llm import LLMBase

T = TypeVar("T")


from html import escape
from typing import Any, Generic, List, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


def format_value(value: Any, indent: int = 0) -> str:
    if isinstance(value, BaseModel):
        return format_pydantic_model(value, indent)
    elif isinstance(value, dict):
        return format_dict(value, indent)
    elif isinstance(value, list):
        return format_list(value, indent)
    else:
        return escape(str(value))


def format_pydantic_model(model: BaseModel, indent: int = 0) -> str:
    return format_dict(model.model_dump(), indent)


def format_dict(d: dict, indent: int = 0) -> str:
    indent_str = "  " * indent
    formatted = "{\n"
    for key, value in d.items():
        formatted += f"{indent_str}  {key}: {format_value(value, indent + 1)}\n"
    formatted += f"{indent_str}}}"
    return formatted


def format_list(l: list, indent: int = 0) -> str:
    indent_str = "  " * indent
    formatted = "[\n"
    for item in l:
        formatted += f"{indent_str}  {format_value(item, indent + 1)},\n"
    formatted += f"{indent_str}]"
    return formatted


class DiscussionResult(BaseModel, Generic[T]):
    query: str
    thoughts: List[Any]
    answer: T

    def __repr__(self) -> str:
        css = """
        <style>
            .discussion-result {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 20px auto;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 10px;
                background-color: #f8f9fa;
                color: #333;
            }
            .query {
                background-color: #e9ecef;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                font-weight: bold;
            }
            .thought-process {
                margin-left: 20px;
            }
            .turn {
                background-color: #ffffff;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .turn.final {
                background-color: #d4edda;
            }
            .turn-content {
                margin-left: 20px;
            }
            .label {
                font-weight: bold;
                color: #495057;
                margin-right: 10px;
            }
            .ida-label {
                color: #0056b3;
            }
            .llma-label {
                color: #28a745;
            }
            pre {
                white-space: pre-wrap;
                word-wrap: break-word;
                background-color: #f1f3f5;
                padding: 10px;
                border-radius: 5px;
                max-height: 300px;
                overflow-y: auto;
            }
            .final-answer {
                background-color: #d4edda;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
            }
        </style>
        """

        html = f"{css}<div class='discussion-result'>"
        html += f"<h2>Discussion Result</h2>"
        html += f"<div class='query'>{escape(self.query)}</div>"
        html += f"<h3>Thought Process:</h3>"
        html += f"<div class='thought-process'>"

        for turn in self.thoughts:
            turn_class = "turn final" if turn.is_final else "turn"
            html += f"""
                <div class='{turn_class}'>
                    <strong>Iteration {turn.iteration}:</strong>
                    <div class='turn-content'>
                        <p><span class='label ida-label'>IDA:</span> {escape(turn.brain_thought)}</p>
                        <p><span class='label llma-label'>LLMA:</span> <pre>{format_value(turn.llm_response)}</pre></p>
                    </div>
                </div>
            """

        html += f"""
            </div>
            <div class='final-answer'>
                <strong>Final Answer:</strong> <pre>{format_value(self.answer)}</pre>
            </div>
        </div>
        """

        return html

    def _repr_html_(self) -> str:
        return self.__repr__()


class MultiAgentBase(Generic[T]):
    def __init__(self, llm: LLMBase):
        self.llm = llm

    def run(self, query: str) -> DiscussionResult[T]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # We're likely in a Jupyter environment
            return asyncio.ensure_future(self._run_async_impl(query))
        else:
            return loop.run_until_complete(self._run_async_impl(query))

    def run_async(self, query: str) -> Future:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        future = loop.create_future()
        asyncio.ensure_future(self._run_async_and_set_future(query, future))
        return future

    async def _run_async_and_set_future(self, query: str, future: Future):
        try:
            result = await self._run_async_impl(query)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

    async def _run_async_impl(self, query: str) -> DiscussionResult[T]:
        raise NotImplementedError("Subclasses must implement this method")
