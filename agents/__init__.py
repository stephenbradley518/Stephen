"""Agentic orchestration package (LangGraph).

Wires the deterministic business core (:mod:`app`) into a stateful agentic
workflow:

    roi_node  ->  decision_node  --(ROI < threshold?)-->  remediation_node  ->  END
                                  \\--(healthy)-----------------------------> END

* :mod:`agents.state`             – the shared graph state (``OrchestrationState``).
* :mod:`agents.decision_agent`    – evaluates ROI against the threshold.
* :mod:`agents.remediation_agent` – drafts a mock supplier outreach email.
* :mod:`agents.graph`             – builds and runs the LangGraph ``StateGraph``.
"""
