from enum import Enum
from typing import Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


class ContractQuestionAnswer(BaseModel):
    text: str = Field(..., description="質問に対する回答文") 


class ContractQAAssistant:
    LLM_MODEL: str = "gpt-4o"

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
                model=self.LLM_MODEL,
                temperature=0.0
            ).with_structured_output(
                ContractQuestionAnswer
            )

    def run(
            self,
            question: Annotated[str, "ユーザーの質問"],
            contract_file_name: Annotated[str, "契約書タイトル"],
            contract_full_text: Annotated[str, "契約書本文"]) -> ContractQuestionAnswer:
        # TODO: web検索tool利用可能なReActエージェントで定義し直したい
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは検索結果の提示と、検索結果として見つかった契約書の内容を参照して質問に回答するアシスタントです。"
                    "以下のことを実行してください。\n"
                    "1. はじめに契約書ファイル名に記載された文書が検索によって見つかったことを提示\n"
                    "2. ユーザー質問に対して契約書本文を参照して回答\n"
                ),
                (
                    "human",
                    f"ユーザー質問: {question}"
                    f"契約書ファイル名: {contract_file_name}\n"
                    f"契約書本文: {contract_full_text}"
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"question": question})
