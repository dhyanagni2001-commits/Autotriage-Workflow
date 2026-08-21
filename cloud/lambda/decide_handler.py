"""
Decide Lambda: runs the agent's rule-based routing logic plus the one
real LLM call in the pipeline (drafting a reply — see
agent/README.md's decision table and agent/tools.py::draft_reply).

Deliberately thin: all the actual logic lives in agent.nodes.decide so
it's identical between local `python -m agent.graph` runs and this
Lambda. Requires OPENAI_API_KEY in the Lambda's environment (or the LLM
call falls back to a deterministic template — see tools.py).

Step Functions gives this state fewer retries (2, vs. 3 elsewhere) since
an LLM timeout is less likely to be transient than a network blip.
"""
from agent.nodes import decide


def handler(event, context):
    return decide(event)
