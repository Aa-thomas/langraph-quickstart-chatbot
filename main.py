from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from dotenv import load_dotenv

load_dotenv()


# Store the messages
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Create the graph builder
graph_builder = StateGraph(State)


# Define the human assistance tool
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


# Define the tools
tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]

# Define the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


# Define the chatbot function
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


# Add the chatbot node
graph_builder.add_node("chatbot", chatbot)

# Add the tool node
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add the conditional edges
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Add the memory to save chat history
memory = MemorySaver()

# Compile the graph
graph = graph_builder.compile(checkpointer=memory)


# Stream the graph updates
def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()


# Main loop
if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except Exception as e:
            # Better error handling
            print(f"An error occurred: {str(e)}")
            break
