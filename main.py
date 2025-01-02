from enum import Enum

import click
from dotenv import load_dotenv
from langchain.output_parsers.enum import EnumOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from tools.contract_qa_assistant import ContractQAAssistant
from tools.legal_qa_assistant import QAAssistant
from tools.search_contract_files import DocumentSearcher
from tools.search_keyword_creator import SearchKeywordCreator
from tools.web_search import WebSearch

load_dotenv()


class ToolEnum(str, Enum):
    QA_ASSISTANT = "qa_assistant"
    WEB_SEARCH = "web_search"
    SEARCH_CONTRACT_FILES = "search_contract_files"


class SelectResult(BaseModel):
    selected_tool: ToolEnum = Field(..., description="Selected tool.")


class ContractAgentState(BaseModel):
    question: str = Field(..., description="ユーザーからの質問")
    selected_tool: str = Field(
        default="",
        description="Selected tool."
    )
    contract_file_content: str = Field(
        default="",
        description="契約書の内容"
    )
    answer: str = Field(default="", description="生成された回答")


class Selector:
    LLM_MODEL: str = "gpt-4o"

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=self.LLM_MODEL,
            temperature=0.0
        )
        self.parser = EnumOutputParser(enum=ToolEnum)

    def run(self, question: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは法律に関する質問回答を行うAIです。"
                    "ユーザーの質問に応じて行うべきタスクを選択してください。"
                    "出力は必ずツール名だけを出力してください。"
                ),
                (
                    "human",
                    "ユーザーの入力に基づいて、次のアクションを行うためのツールを選択してください:\n"
                    "- qa_assistant: 法律に関する質問について回答するツール\n"
                    "- web_search: web検索を行い収集した情報で回答するツール\n"
                    "- search_contract_files: 契約書ファイルを検索して回答するツール\n"
                    f"ユーザー質問: {question}"
                ),
            ]
        )
        chain = prompt | self.llm | self.parser
        return chain.invoke({"question": question})


class ContractAgent:
    def __init__(self) -> None: 
        self.qa_assistant = QAAssistant()
        self.web_search = WebSearch()
        self.selector = Selector()
        self.search_keyword_creator = SearchKeywordCreator()
        self.document_searcher = DocumentSearcher()
        self.contract_qa_assistant = ContractQAAssistant()

    @property
    def graph(self) -> StateGraph:
        return self._create_graph()

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(ContractAgentState)
        workflow.add_node("selector", self._select_tool)
        workflow.add_node("qa_assistant", self._qa_assistant)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("search_contract_files", self._search_contracts)
        workflow.add_conditional_edges(
            "selector",
            self._routing
        )
        workflow.set_entry_point("selector")
        workflow.add_edge("qa_assistant", END)
        workflow.add_edge("web_search", END)
        workflow.add_edge("search_contract_files", END)
        return workflow.compile()

    def _routing(self, state: ContractAgentState) -> ContractAgentState:
        match state.selected_tool:
            case ToolEnum.QA_ASSISTANT:
                return "qa_assistant"
            case ToolEnum.WEB_SEARCH:
                return "web_search"
            case ToolEnum.SEARCH_CONTRACT_FILES:
                return "search_contract_files"
            case _:  # noqa
                raise ValueError(f"Invalid tool: {state.selected_tool}")

    def _qa_assistant(self, state: ContractAgentState) -> dict[str, str]:
        result = self.qa_assistant.run(state.question)
        return {"answer": result.text}

    def _web_search(self, state: ContractAgentState) -> dict[str, str]:
        result = self.web_search.run(state.question)
        return {"answer": result}

    def _search_contracts(self, state: ContractAgentState) -> dict[str, str]:
        search_keyword = self.search_keyword_creator.run(state.question)
        search_result = self.document_searcher.search(search_keyword.text)
        result = self.contract_qa_assistant.run(
            question=state.question,
            contract_title=search_result.file_name,
            contract_full_text=search_result.full_text
        )
        return {"answer": result.text}

    def _select_tool(self, state: ContractAgentState) -> dict:
        result = self.selector.run(state.question)
        return {"selected_tool": result}

    def run(self, question: str) -> str:
        initial_state = ContractAgentState(question=question)
        final_state = self.graph.invoke(initial_state)
        return final_state["answer"]


@click.command()
@click.option(
    '--question',
    type=str,
    prompt='Please enter your question:',
    help='Question to AI agent.'
)
def main(question: str) -> None:
    agent = ContractAgent()
    final_output = agent.run(question=question)
    print(final_output)


if __name__ == "__main__":
    main()
