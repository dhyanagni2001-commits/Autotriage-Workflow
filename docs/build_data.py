"""Regenerate docs/data.json (the demo site's data) from the trained
models and results/*.json. Run from the repo root after training + evals:
    python docs/build_data.py
"""
import json
import logging

import yaml

from agent.graph import build_graph

logging.disable(logging.CRITICAL)
graph = build_graph()
runs = []
for path, set_name in [("eval/labeled_tickets.jsonl", "dev"), ("eval/heldout_tickets.jsonl", "heldout")]:
    for line in open(path):
        t = json.loads(line)
        r = graph.invoke({"ticket_text": t["ticket_text"], "amount_usd": t["amount_usd"]})
        arts = r.get("retrieved_articles") or []
        runs.append({
            "set": set_name, "slice": t["slice"], "text": t["ticket_text"], "amount": t["amount_usd"],
            "expected": t["expected_action"], "action": r["action"], "intent": r.get("intent"),
            "category": r.get("category"), "confidence": round(r.get("confidence", 0), 3),
            "trace": r.get("reasoning_trace", []), "article": arts[0]["id"] if arts else None,
            "tools": [c["tool"] for c in r.get("tool_calls", [])],
        })

agent_cfg = yaml.safe_load(open("config.yaml"))["agent"]
results = {}
for k in ["retrieval", "classifier", "e2e_labeled", "e2e_labeled_before", "e2e_heldout", "e2e_heldout_before"]:
    results[k] = json.load(open(f"results/{k}.json"))
    results[k].pop("misses", None)

json.dump({
    "runs": runs,
    "results": results,
    "config": {"threshold": agent_cfg["confidence_threshold"],
               "refund_limit": agent_cfg["auto_approve_refund_threshold_usd"],
               "risk_patterns": agent_cfg["risk_patterns"]},
}, open("docs/data.json", "w"))
print(f"wrote docs/data.json ({len(runs)} runs)")
