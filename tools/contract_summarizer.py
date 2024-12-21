from enum import Enum
from typing import Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ContractSummarized(BaseModel):
    text: str = Field(..., description="要約された契約書本文")


class ContractSummarizer:
    LLM_MODEL: str = "gpt-4o"

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
                model=self.LLM_MODEL,
                temperature=0.0
            ).with_structured_output(
                ContractSummarized
            )

    def run(self, contract_text: Annotated[str, "契約書本文"]) -> ContractSummarized:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは契約書文書の翻訳を行うAIです。",
                ),
                (
                    "human",
                    "契約書文書の翻訳を行ってください。\n"
                    f"契約書本文: {contract_text}"
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"contract_text": contract_text})
