"""
Agent Pipeline — orchestrates classification → retrieval → response
for each support ticket, and provides batch processing.
"""

import pandas as pd
from typing import Callable, Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from config import INPUT_CSV, SAMPLE_CSV, OUTPUT_CSV
from retriever import get_relevant_context
from chains import (
    classify_ticket,
    generate_response,
    generate_escalation,
)


@dataclass
class TicketResult:
    """Holds the complete result for one support ticket."""
    issue: str
    subject: str
    company: str
    status: str            # replied | escalated
    request_type: str      # product_issue | feature_request | bug | invalid
    product_area: str
    response: str
    justification: str


def process_ticket(
    issue: str,
    subject: str,
    company: str,
    log_cb: Optional[Callable[[str], None]] = None,
) -> TicketResult:
    """
    Process a single support ticket through the full pipeline:
      1. Classify (type, area, escalation decision)
      2. If escalated → generate escalation response
      3. If replied  → retrieve context → generate grounded response
    """

    def _log(msg: str):
        if log_cb:
            log_cb(msg)

    # ── Step 1: Classify ──────────────────────────────────────────────
    _log("Classifying ticket…")
    classification = classify_ticket(issue, subject, company)

    effective_company = company or ""
    if (not effective_company or effective_company.strip().lower() == "none") and classification.inferred_company:
        effective_company = classification.inferred_company
        _log(f"Inferred company: {effective_company}")

    _log(
        f"Classification: type={classification.request_type}, "
        f"area={classification.product_area}, "
        f"escalate={classification.should_escalate}"
    )

    # ── Step 2: Escalate or Respond ───────────────────────────────────
    if classification.should_escalate:
        _log("⚠️  Escalating to human agent…")
        esc = generate_escalation(
            issue=issue,
            subject=subject,
            company=effective_company,
            escalation_reason=classification.escalation_reason or "Requires human review",
        )
        return TicketResult(
            issue=issue,
            subject=subject or "",
            company=effective_company,
            status="escalated",
            request_type=classification.request_type,
            product_area=esc.product_area or classification.product_area,
            response=esc.response,
            justification=esc.justification,
        )

    # ── Step 3: Retrieval ─────────────────────────────────────────────
    _log(f"Retrieving context from '{effective_company}' index…")
    try:
        context = get_relevant_context(issue, effective_company)
        _log(f"Retrieved context ({len(context)} chars).")
    except Exception as e:
        _log(f"Retrieval error: {e} — falling back to empty context.")
        context = ""

    # ── Step 4: RAG Response generation ───────────────────────────────
    _log("Generating response…")
    resp = generate_response(
        issue=issue,
        subject=subject,
        company=effective_company,
        request_type=classification.request_type,
        product_area=classification.product_area,
        context=context,
    )

    return TicketResult(
        issue=issue,
        subject=subject or "",
        company=effective_company,
        status="replied",
        request_type=classification.request_type,
        product_area=classification.product_area,
        response=resp.response,
        justification=resp.justification,
    )


def load_tickets(csv_path=None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read the input CSV and return a list of ticket dicts."""
    path = csv_path or INPUT_CSV
    df = pd.read_csv(path, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    records = df.to_dict(orient="records")
    if limit is not None:
        return records[:limit]
    return records


def process_all_tickets(
    csv_path=None,
    output_path=None,
    progress_cb: Optional[Callable[[int, int, TicketResult], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    limit: Optional[int] = None,
) -> List[TicketResult]:
    """
    Batch-process every ticket in the input CSV.

    Args:
        csv_path:     override input CSV path
        output_path:  override output CSV path
        progress_cb:  called after each ticket with (index, total, result)
        log_cb:       called with status messages
        limit:        process only this many tickets
    """
    tickets = load_tickets(csv_path, limit=limit)
    total = len(tickets)
    results: List[TicketResult] = []

    for i, ticket in enumerate(tickets):
        if log_cb:
            log_cb(f"\n{'='*50}")
            log_cb(f"Ticket {i+1}/{total}")
            log_cb(f"Company: {ticket.get('company', 'None')}")
            log_cb(f"Subject: {ticket.get('subject', '')[:60]}")

        result = process_ticket(
            issue=ticket.get("issue", ""),
            subject=ticket.get("subject", ""),
            company=ticket.get("company", ""),
            log_cb=log_cb,
        )
        results.append(result)

        if progress_cb:
            progress_cb(i + 1, total, result)

    # ── Write output CSV ──────────────────────────────────────────────
    out = output_path or OUTPUT_CSV
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    cols = ["issue", "subject", "company", "response", "product_area",
            "status", "request_type", "justification"]
    df = df[cols]
    df.to_csv(out, index=False)

    if log_cb:
        log_cb(f"\n✅ Results written to {out}")
        replied = sum(1 for r in results if r.status == "replied")
        escalated = sum(1 for r in results if r.status == "escalated")
        log_cb(f"Summary: {replied} replied, {escalated} escalated, {total} total")

    return results
