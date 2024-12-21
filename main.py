from enum import Enum

import click
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from tools.contract_summarizer import ContractSummarizer
from tools.legal_qa_assistant import QAAssistant

load_dotenv()


class ToolEnum(str, Enum):
    SUMMARIZER = "summarizer"
    QA_ASSISTANT = "qa_assistant"


class SelectResult(BaseModel):
    selected_tool: ToolEnum = Field(..., description="Selected tool.")


class ContractAgentState(BaseModel):
    question: str = Field(..., description="ユーザーからの質問")
    selected_tool: str = Field(
        default="",
        description="Selected tool."
    )
    answer: str = Field(default="", description="生成された回答")


class Selector:
    LLM_MODEL: str = "gpt-4o"

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=self.LLM_MODEL,
            temperature=0.0
        )

    def run(self, question: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは法律に関する質問回答を行うAIです。"
                    "ユーザーの質問に応じて行うべきタスクを選択してください。",
                ),
                (
                    "human",
                    "ユーザーの入力に基づいて、次のアクションを行うためのツールを選択してください:\n"
                    "- summarizer\n"
                    "- qa_assistant\n"
                    f"ユーザー質問: {question}"
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question})


class ContractAgent:
    def __init__(self) -> None: 
        self.summarizer = ContractSummarizer()
        self.qa_assistant = QAAssistant()
        self.selector = Selector()

    @property
    def graph(self) -> StateGraph:
        return self._create_graph()

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(ContractAgentState)
        workflow.add_node("selector", self._select_tool)
        workflow.add_node("summarizer", self._summarize_contract)
        workflow.add_node("qa_assistant", self._qa_assistant)
        workflow.add_conditional_edges(
            "selector",
            lambda state: state.selected_tool == ToolEnum.SUMMARIZER,
            {
                True: "summarizer",
                False: "qa_assistant",
            }
        )
        workflow.set_entry_point("selector")
        workflow.add_edge("summarizer", END)
        workflow.add_edge("qa_assistant", END)
        return workflow.compile()

    def _summarize_contract(self, state: ContractAgentState) -> dict[str, str]:
        result = self.summarizer.run(state.question)
        return {"answer": result.text}

    def _qa_assistant(self, state: ContractAgentState) -> dict[str, str]:
        result = self.qa_assistant.run(state.question)
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
