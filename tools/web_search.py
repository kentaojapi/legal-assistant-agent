from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


class WebSearch:
    LLM_MODEL: str = "gpt-4o"

    def __init__(self) -> None:
        self.llm = ChatOpenAI(model=self.LLM_MODEL, temperature=0.0)
        self.retriever = TavilySearchResults(max_results=2)

    def run(self, query: Annotated[str, "ユーザーの質問"]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはWeb検索を行うAIです。"
                    "ユーザーの質問に対してWeb検索した内容を利用して適切な回答を返してください。",
                ),
                (
                    "human",
                    "web検索結果: {search_results}\n"
                    f"ユーザー質問: {query}"
                ),
            ]
        )
        chain = (RunnablePassthrough.assign(search_results=(lambda x: query) | self.retriever)
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke({"query": query})
