from enum import Enum
from typing import Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


class QuestionAnswer(BaseModel):
    text: str = Field(..., description="質問に対する回答文") 


class QAAssistant:
    LLM_MODEL: str = "gpt-4o"

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
                model=self.LLM_MODEL,
                temperature=0.0
            ).with_structured_output(
                QuestionAnswer
            )

    def run(self, question: Annotated[str, "ユーザーの質問"]) -> QuestionAnswer:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは法律に関する質問回答を行うAIです。",
                ),
                (
                    "human",
                    "以下のユーザーからの質問に日本法の観点から回答してください\n"
                    f"ユーザー質問: {question}"
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"question": question})
