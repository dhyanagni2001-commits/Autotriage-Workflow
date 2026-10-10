"""
End-to-end pipeline eval: runs the full agent graph over a hand-labeled set
of tickets and reports decision accuracy against the expected action, the
safety-critical error rates, per-slice accuracy, and per-ticket latency.

The headline safety number is the *unsafe auto-resolve rate*: of tickets a
human says must NOT be auto-resolved, how many did the agent auto-resolve
anyway (i.e. issued a refund it shouldn't have). This answers "does the
agent know when NOT to act?"

Usage:
    python -m eval.run_eval_suite --test-file eval/labeled_tickets.jsonl
"""
import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from agent.graph import build_graph

ACTIONS = ["auto_resolve", "draft_hold", "escalate"]


def load_labeled_tickets(path):
    """Expected format per line:
    {"ticket_text": "...", "amount_usd": 12.99, "expected_action": "auto_resolve", "slice": "refund_small"}
    """
    return [json.loads(l) for l in open(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="eval/labeled_tickets.jsonl")
    parser.add_argument("--out", default="results/e2e.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every misrouted ticket")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    tickets = load_labeled_tickets(args.test_file)
    graph = build_graph()
    graph.invoke({"ticket_text": "warmup", "amount_usd": None})  # load models outside the timer

    confusion = defaultdict(Counter)
    by_slice = defaultdict(lambda: [0, 0])
    latencies, misses = [], []

    for t in tickets:
        start = time.perf_counter()
        result = graph.invoke({"ticket_text": t["ticket_text"], "amount_usd": t.get("amount_usd")})
        latencies.append((time.perf_counter() - start) * 1000)
        exp, got = t["expected_action"], result["action"]
        confusion[exp][got] += 1
        by_slice[t.get("slice", "all")][0] += exp == got
        by_slice[t.get("slice", "all")][1] += 1
        if exp != got:
            misses.append({"ticket": t["ticket_text"], "expected": exp, "got": got,
                           "intent": result.get("intent"), "confidence": round(result.get("confidence", 0), 3)})

    n = len(tickets)
    correct = sum(confusion[a][a] for a in ACTIONS)
    not_auto = sum(sum(confusion[a].values()) for a in ACTIONS if a != "auto_resolve")
    unsafe = sum(confusion[a]["auto_resolve"] for a in ACTIONS if a != "auto_resolve")
    should_auto = sum(confusion["auto_resolve"].values())
    got = Counter()
    for row in confusion.values():
        got.update(row)

    metrics = {
        "n": n,
        "decision_accuracy": correct / n,
        "unsafe_auto_resolve_rate": unsafe / not_auto if not_auto else 0.0,
        "auto_resolve_recall": confusion["auto_resolve"]["auto_resolve"] / should_auto if should_auto else None,
        "auto_resolve_rate": got["auto_resolve"] / n,
        "escalation_rate": got["escalate"] / n,
        "latency_ms": {"p50": float(np.percentile(latencies, 50)), "p95": float(np.percentile(latencies, 95))},
        "by_slice": {s: c / tot for s, (c, tot) in sorted(by_slice.items())},
        "confusion": {e: dict(confusion[e]) for e in ACTIONS},
        "misses": misses,
    }

    print(f"n = {n} tickets")
    print(f"decision accuracy:         {metrics['decision_accuracy']:.1%}")
    print(f"unsafe auto-resolve rate:  {metrics['unsafe_auto_resolve_rate']:.1%}  ({unsafe}/{not_auto})")
    print(f"auto-resolve recall:       {metrics['auto_resolve_recall']:.1%}")
    print(f"auto-resolve / escalation: {metrics['auto_resolve_rate']:.1%} / {metrics['escalation_rate']:.1%}")
    print(f"latency p50 / p95:         {metrics['latency_ms']['p50']:.1f} / {metrics['latency_ms']['p95']:.1f} ms")
    print("\nconfusion (rows = expected, cols = predicted):")
    print(f"{'':<14}" + "".join(f"{a:>14}" for a in ACTIONS))
    for e in ACTIONS:
        print(f"{e:<14}" + "".join(f"{confusion[e][a]:>14}" for a in ACTIONS))
    print("\nper-slice accuracy:")
    for s, acc in metrics["by_slice"].items():
        print(f"  {s:<14}{acc:>7.1%}")
    if args.verbose:
        for m in misses:
            print("MISS", m)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
