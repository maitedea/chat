from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, add_messages


### ------- Graph function definition ------- ###

def build_graph(llm=None, prompt: str = "Be a helpful assistant", tools: list = None):

    ### ------- State definition ------- ###

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    ### ------- Nodes definition ------- ###

    system_prompt = prompt

    # Define the function that calls the model
    def call_model(state, config):
        model = llm
        if tools:
            model = model.bind_tools(tools)

        messages = state["messages"]
        messages = [{"role": "system", "content": system_prompt}] + messages
        response = model.invoke(messages)
        return {"messages": [response]}

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Add the agent node
    workflow.add_node("agent", call_model)

    # Set the entrypoint as `agent`
    workflow.set_entry_point("agent")

    if tools:
        # Define the tool node if tools are provided
        tool_node = ToolNode(tools)
        workflow.add_node("action", tool_node)

        # Add conditional edges
        def should_continue(state):
            messages = state["messages"]
            last_message = messages[-1]
            if not last_message.tool_calls:
                return "end"
            else:
                return "continue"

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )

        # Add edge from `action` back to `agent`
        workflow.add_edge("action", "agent")
    else:
        # If no tools, directly end after `agent`
        workflow.add_edge("agent", END)

    # Finally, we compile it!
    return workflow.compile()