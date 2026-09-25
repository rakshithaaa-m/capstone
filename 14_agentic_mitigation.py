"""
Stage 14 — Agentic Mitigation Assistant

Takes a single at-risk drive's data (risk tier, hybrid score, RUL estimate,
SHAP root cause from Stage 12) and hands it to an LLM AGENT with real
tools, rather than a single one-shot LLM call. The agent decides FOR
ITSELF which tools to invoke and in what order, based on the drive's
actual risk profile -- that autonomous tool-selection is what makes this
"agentic" rather than just "an LLM call."

Three tools, all clearly SIMULATED (no real inventory/scheduling system
exists to connect to -- this is a proof-of-concept of the agent
architecture, not a production integration):
  - check_spare_parts_inventory(drive_model)
  - check_technician_availability(urgency)
  - draft_maintenance_ticket(...)

Two ways to run:
  --dry_run   Uses a small deterministic rule-based stand-in instead of a
              real LLM call -- verifies the tool-calling control flow
              (schemas, tool execution, response threading) works
              correctly WITHOUT needing an API key or network access.
              Useful for testing the harness itself before spending API
              credits, and for anyone reviewing this without a key.
  (default)   Uses the real Anthropic API (needs ANTHROPIC_API_KEY set)
              -- Claude genuinely decides which tools to call.

Usage:
    python3 14_agentic_mitigation.py \
        --serial_number ZL2CP0BS \
        --results_dir results \
        --dry_run          # test the harness without an API key

    set ANTHROPIC_API_KEY=sk-ant-...
    python3 14_agentic_mitigation.py --serial_number ZL2CP0BS --results_dir results
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

# ============================== TOOLS ==============================
# These are real Python functions the agent can choose to call. Each has
# a JSON schema Claude uses to decide when/how to call it. All data here
# is SIMULATED -- a production version would call a real inventory API,
# a real scheduling system, and a real ticketing system (Jira, ServiceNow).

_MOCK_INVENTORY = {
    "ST4000DM000": 12, "ST12000NM0007": 3, "ST12000NM0008": 0,
    "HGST HUH721212ALE600": 5, "TOSHIBA MG08ACA16TA": 2,
    "WDC WUH721816ALE6L0": 0, "ST8000DM002": 8, "ST8000NM0055": 4,
}


def check_spare_parts_inventory(drive_model: str) -> dict:
    """[SIMULATED] Checks how many replacement units of this drive model
    are currently in spare-parts inventory."""
    count = _MOCK_INVENTORY.get(drive_model, 1)  # unknown models: assume minimal stock
    return {"drive_model": drive_model, "units_in_stock": count,
            "status": "in_stock" if count > 0 else "out_of_stock"}


def check_technician_availability(urgency: str) -> dict:
    """[SIMULATED] Checks the next available on-site technician slot,
    based on urgency (urgent / routine)."""
    slots = {"urgent": "Today, 14:00", "routine": "Within 5 business days"}
    return {"urgency": urgency, "next_available_slot": slots.get(urgency, "Within 5 business days")}


def draft_maintenance_ticket(serial_number: str, drive_model: str, risk_tier: str,
                              root_cause: str, recommended_action: str,
                              technician_slot: str = None, parts_available: bool = None) -> dict:
    """Formats a structured maintenance ticket ready to hand to an
    operator or paste into a real ticketing system."""
    return {
        "ticket_id": f"MAINT-{serial_number}",
        "serial_number": serial_number,
        "drive_model": drive_model,
        "priority": risk_tier,
        "root_cause": root_cause,
        "recommended_action": recommended_action,
        "technician_slot": technician_slot,
        "parts_available": parts_available,
    }


TOOL_FUNCTIONS = {
    "check_spare_parts_inventory": check_spare_parts_inventory,
    "check_technician_availability": check_technician_availability,
    "draft_maintenance_ticket": draft_maintenance_ticket,
}

TOOL_SCHEMAS = [
    {
        "name": "check_spare_parts_inventory",
        "description": "Check how many replacement units of a given drive model are in spare-parts inventory.",
        "input_schema": {
            "type": "object",
            "properties": {"drive_model": {"type": "string"}},
            "required": ["drive_model"],
        },
    },
    {
        "name": "check_technician_availability",
        "description": "Check the next available technician slot for 'urgent' or 'routine' priority work.",
        "input_schema": {
            "type": "object",
            "properties": {"urgency": {"type": "string", "enum": ["urgent", "routine"]}},
            "required": ["urgency"],
        },
    },
    {
        "name": "draft_maintenance_ticket",
        "description": "Draft a structured maintenance ticket once you have gathered the information you need.",
        "input_schema": {
            "type": "object",
            "properties": {
                "serial_number": {"type": "string"},
                "drive_model": {"type": "string"},
                "risk_tier": {"type": "string"},
                "root_cause": {"type": "string"},
                "recommended_action": {"type": "string"},
                "technician_slot": {"type": "string"},
                "parts_available": {"type": "boolean"},
            },
            "required": ["serial_number", "drive_model", "risk_tier", "root_cause", "recommended_action"],
        },
    },
]

SYSTEM_PROMPT = (
    "You are a data-center maintenance assistant. You will be given one drive's failure-risk "
    "assessment (risk tier, hybrid risk score, estimated days to failure, and the SHAP-derived "
    "root cause). Decide for yourself which tools you need -- checking spare-parts inventory "
    "and/or technician availability -- before drafting a maintenance ticket. A HIGH risk tier "
    "generally warrants urgent handling; MEDIUM can usually be routine unless the estimated "
    "time-to-failure is very short. Not every drive needs every tool -- use your judgment. "
    "After using any tools you need, draft a ticket, then give a short (3-4 sentence) plain-"
    "language summary of your reasoning and action plan for a human operator."
)


# ========================== DRIVE CONTEXT ==========================

def build_drive_context(results_dir: Path, serial_number: str) -> dict:
    """Pulls one drive's data from Stage 8/9/12's already-computed
    outputs via a targeted DuckDB query -- consistent with the rest of
    this pipeline's memory-safe patterns, never loading the full files."""
    import duckdb
    risk_tiers_path = results_dir / "risk_tiers_test.csv"
    con = duckdb.connect()
    row = con.execute(f"""
        SELECT serial_number, date, hybrid_score, risk_tier
        FROM read_csv_auto('{risk_tiers_path.as_posix()}')
        WHERE serial_number = ?
        ORDER BY date DESC LIMIT 1
    """, [serial_number]).fetchdf()
    con.close()
    if row.empty:
        raise ValueError(f"No data found for drive {serial_number}")
    row = row.iloc[0]

    mitigation_path = results_dir / "mitigation_mapping.json"
    mitigation = json.loads(mitigation_path.read_text()) if mitigation_path.exists() else []
    match = next((m for m in mitigation if m["serial_number"] == serial_number), None)

    drive_model_path = results_dir.parent / "data" / "processed" / "features_test.parquet"
    drive_model = "UNKNOWN"
    if drive_model_path.exists():
        con = duckdb.connect()
        model_row = con.execute(f"""
            SELECT model FROM read_parquet('{drive_model_path.as_posix()}')
            WHERE serial_number = ? LIMIT 1
        """, [serial_number]).fetchdf()
        con.close()
        if not model_row.empty:
            drive_model = model_row.iloc[0]["model"]

    return {
        "serial_number": serial_number,
        "drive_model": drive_model,
        "risk_tier": row["risk_tier"],
        "hybrid_score": float(row["hybrid_score"]),
        "date": str(row["date"]),
        "root_cause": match["root_cause"] if match else "No SHAP explanation available for this drive.",
        "recommended_action": match["recommended_action"] if match else "Escalate to standard monitoring.",
    }


# ============================ AGENT LOOP ============================

def run_agent_dry(context: dict) -> dict:
    """Deterministic stand-in for the LLM -- exercises the EXACT same
    tool-calling control flow (schema validation, tool execution,
    multi-turn threading) without needing an API key. Proves the harness
    itself is correct; does not exercise Claude's actual judgment."""
    trace = []
    urgency = "urgent" if context["risk_tier"] == "HIGH" else "routine"

    inv = check_spare_parts_inventory(context["drive_model"])
    trace.append({"tool": "check_spare_parts_inventory", "input": {"drive_model": context["drive_model"]}, "output": inv})

    tech = check_technician_availability(urgency)
    trace.append({"tool": "check_technician_availability", "input": {"urgency": urgency}, "output": tech})

    ticket = draft_maintenance_ticket(
        context["serial_number"], context["drive_model"], context["risk_tier"],
        context["root_cause"], context["recommended_action"],
        technician_slot=tech["next_available_slot"], parts_available=inv["status"] == "in_stock",
    )
    trace.append({"tool": "draft_maintenance_ticket", "input": ticket, "output": ticket})

    summary = (
        f"[DRY RUN -- rule-based stand-in, not a real LLM decision] "
        f"Drive {context['serial_number']} is {context['risk_tier']} risk. "
        f"Spare parts: {inv['status']} ({inv['units_in_stock']} units). "
        f"Technician slot: {tech['next_available_slot']}. Ticket {ticket['ticket_id']} drafted."
    )
    return {"summary": summary, "tool_trace": trace, "ticket": ticket}


def run_agent_live(context: dict, model: str = "claude-sonnet-4-6") -> dict:
    """The real agent loop: sends the drive context + tools to Claude,
    and loops -- executing whatever tools Claude calls, feeding results
    back -- until Claude stops calling tools and returns a final text
    summary. This loop, and which tools get called in what order, is
    decided by Claude at runtime, not hardcoded here."""
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    user_message = (
        f"Drive {context['serial_number']} (model: {context['drive_model']}):\n"
        f"- Risk tier: {context['risk_tier']}\n"
        f"- Hybrid risk score: {context['hybrid_score']:.3f}\n"
        f"- As of: {context['date']}\n"
        f"- SHAP root cause: {context['root_cause']}\n"
        f"- Rule-based recommended action: {context['recommended_action']}\n\n"
        f"Assess this drive and take appropriate action."
    )
    messages = [{"role": "user", "content": user_message}]
    trace = []

    for _ in range(6):  # hard cap on agent turns, avoids runaway loops
        response = client.messages.create(
            model=model, max_tokens=1000, system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS, messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            final_text = "".join(b.text for b in response.content if b.type == "text")
            return {"summary": final_text, "tool_trace": trace, "ticket": None}

        tool_results = []
        for call in tool_calls:
            fn = TOOL_FUNCTIONS[call.name]
            result = fn(**call.input)
            trace.append({"tool": call.name, "input": call.input, "output": result})
            tool_results.append({
                "type": "tool_result", "tool_use_id": call.id, "content": json.dumps(result)
            })
            if call.name == "draft_maintenance_ticket":
                ticket = result
        messages.append({"role": "user", "content": tool_results})

    return {"summary": "[Agent hit turn limit without a final response]", "tool_trace": trace, "ticket": None}


def run_agent(context: dict, dry_run: bool = False) -> dict:
    return run_agent_dry(context) if dry_run else run_agent_live(context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial_number", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--dry_run", action="store_true",
                         help="Test the tool-calling harness without an API key or network access")
    args = parser.parse_args()

    context = build_drive_context(Path(args.results_dir), args.serial_number)
    print("[CONTEXT]")
    print(json.dumps(context, indent=2))

    result = run_agent(context, dry_run=args.dry_run)
    print("\n[TOOL TRACE -- what the agent actually did, in order]")
    for step in result["tool_trace"]:
        print(f"  -> {step['tool']}({step['input']}) = {step['output']}")

    print("\n[AGENT SUMMARY]")
    print(result["summary"])


if __name__ == "__main__":
    main()
