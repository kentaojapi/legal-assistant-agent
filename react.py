from typing import Annotated

import click
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools.legal_qa_assistant import QAAssistant
from tools.search_contract_files import DocumentSearcher, SearchResult


@tool
def search_contract_files(
        search_keyword: Annotated[str, "検索キーワード"]) -> list[SearchResult]:
    """契約書を検索するツール"""
    return DocumentSearcher().search(search_keyword)


@tool
def legal_qa_assistant(
        question: Annotated[str, "質問内容"]) -> str:
    """法律に関する質問回答を行うツール"""
    return QAAssistant().run(question)


class ReactAgent:
    def __init__(self) -> None:
        self.model = ChatOpenAI(model="gpt-4o")
        self.tools = [
            TavilySearchResults(max_results=2),
            search_contract_files,
            legal_qa_assistant
        ]

    @property
    def agent(self):
        return create_react_agent(self.model, self.tools, debug=True)

    def invoke(self, question: str) -> dict:
        return self.agent.invoke(
            {"messages": [("human", question)]}
        )


@click.command()
@click.option(
    '--question',
    type=str,
    prompt='Please enter your question:',
    help='Question to AI agent.'
)
def main(question: str) -> None:
    agent = ReactAgent()
    result = agent.invoke(question)
    print("")
    print("Final output:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
