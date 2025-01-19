import getpass
import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langchain_core.tools import tool

# Load environment variables
load_dotenv()


# Define state type
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize tools
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


tavily_tool = TavilySearchResults(max_results=2)
tools = [human_assistance, tavily_tool]


# Initialize LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

# Initialize memory saver
memory = MemorySaver()


# Define chatbot function
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

# Add tool node
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# # Add edges
# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)

# Add conditional edges
# # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    "chatbot",
    tools_condition,
    {"tools": "tools", END: END},
)


# Stream function
def stream_graph_updates(user_input: str):
    user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
    config = {"configurable": {"thread_id": "1"}}

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()


# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")  # After tool execution → go back to chatbot
graph_builder.add_edge(START, "chatbot")  # Start the graph at the chatbot

# Compile graph
graph = graph_builder.compile(checkpointer=memory)


# Visualization (optional)
try:
    from IPython.display import Image

    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass


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
