"""
End-to-end pipeline eval: runs the full agent graph over a labeled test
set of tickets and reports auto-resolve rate, escalation rate, and
"correctness" of the decision against a human-labeled expected action.

This is the number that answers: "does the agent know when NOT to act?"
which is the actual interesting claim of this project.

Usage:
    python eval/run_eval_suite.py --test-file eval/labeled_tickets.jsonl
"""
import argparse
import json
from collections import Counter

from agent.graph import build_graph


def load_labeled_tickets(path):
    """Expected format per line:
    {"ticket_text": "...", "amount_usd": 12.99, "expected_action": "auto_resolve"}
    """
    return [json.loads(l) for l in open(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="eval/labeled_tickets.jsonl")
    args = parser.parse_args()

    tickets = load_labeled_tickets(args.test_file)
    graph = build_graph()

    action_counts = Counter()
    correct = 0

    for t in tickets:
        result = graph.invoke({"ticket_text": t["ticket_text"], "amount_usd": t.get("amount_usd")})
        action_counts[result["action"]] += 1
        if result["action"] == t.get("expected_action"):
            correct += 1

    n = len(tickets)
    print(f"n = {n} tickets")
    print("action distribution:", dict(action_counts))
    print(f"auto-resolve rate: {action_counts['auto_resolve'] / n:.2%}")
    print(f"escalation rate:   {action_counts['escalate'] / n:.2%}")
    if any("expected_action" in t for t in tickets):
        print(f"decision accuracy vs. expected_action: {correct / n:.2%}")


if __name__ == "__main__":
    main()
