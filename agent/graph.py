"""
Builds the AutoTriage LangGraph state machine:
    retrieve -> classify -> decide -> act

Usage:
    from agent.graph import build_graph
    graph = build_graph()
    result = graph.invoke({"ticket_text": "...", "amount_usd": 12.99})
"""
from langgraph.graph import END, StateGraph

from agent.nodes import act, classify, decide, retrieve
from agent.state import TicketState


def build_graph():
    workflow = StateGraph(TicketState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("classify", classify)
    workflow.add_node("decide", decide)
    workflow.add_node("act", act)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "classify")
    workflow.add_edge("classify", "decide")
    workflow.add_edge("decide", "act")
    workflow.add_edge("act", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()
    example = {
        "ticket_text": "Hi, I was charged twice for my subscription this month, can you refund one of the charges? Order #48213.",
        "amount_usd": 12.99,
    }
    result = graph.invoke(example)
    print("action:", result["action"])
    print("reasoning:", result["reasoning_trace"])
    print("tool_calls:", result["tool_calls"])
