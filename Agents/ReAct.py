from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """ This function returns addition of two numbers"""
    return a+b

tools = [add]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite").bind_tools(tools)

def model_call(state: AgentState)-> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI Assistant, Please answer my question to the best of your ability."
    )

    response = llm.invoke([system_prompt]+state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    mesages = state["messages"]
    last_message = mesages[-1]
    # print(f"last_message: {last_message}")
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("agent", model_call)
graph.set_entry_point("agent")

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tool_node",
        "end": END
    }
)

graph.add_edge("tool_node","agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 20 + 40")]}
print_stream(app.stream(inputs, stream_mode="values"))
