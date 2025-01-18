import getpass
import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
import json
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv()


# Define state type
class State(TypedDict):
    messages: Annotated[list, add_messages]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


# Initialize tools
tool = TavilySearchResults(max_results=2)
tools = [tool]
tool.invoke("What is a node in langgraph?")


# Initialize LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

# Initialize memory saver
memory = MemorySaver()


# Define chatbot function
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

# Add tool node
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

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


# Stream function
def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()
    snapshot = graph.get_state(config)
    print(snapshot)


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
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break
