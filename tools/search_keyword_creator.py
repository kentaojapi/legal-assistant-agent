from enum import Enum
from typing import Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class SearchKeyword(BaseModel):
    text: str = Field(..., description="全文検索を行うためのキーワード")


class SearchKeywordCreator:
    LLM_MODEL: str = "gpt-4o"

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
                model=self.LLM_MODEL,
                temperature=0.0
            ).with_structured_output(
                SearchKeyword
            )

    def run(self, question: Annotated[str, "ユーザーの質問"]) -> SearchKeyword:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはユーザーの質問文を理解した上で全文検索を行うための質問キーワードを生成するAIです。\n"
                    "Q: 物件売買契約書をください\n"
                    "A: 物件売買契約書\n"
                    "Q: 物件売買契約書で違約金の規定についてどのように書いてありますか？\n"
                    "A: 物件売買契約書 違約金 規定"
                ),
                (
                    "human",
                    f"ユーザーの質問: {question}\n"
                    "A:"
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"question": question})
