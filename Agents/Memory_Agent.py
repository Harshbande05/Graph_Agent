from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)

graph.add_node("process",process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    # print(f"Conv_history: {conversation_history}")
    user_input = input("Enter: ")

with open("logging.txt", 'w') as f:
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"you: {message.content}\n")
        if isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n\n")
    f.write("End of conversation!")

print("Converstation Saved.")