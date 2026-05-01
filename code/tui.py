"""
Textual Terminal UI for the Support Triage Agent.

Provides a rich interactive interface with progress tracking,
live ticket display, and processing controls.
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import (
    Header,
    Footer,
    Static,
    ProgressBar,
    RichLog,
)
from textual.worker import get_current_worker
from textual import work

from agent import process_ticket, load_tickets, TicketResult
from indexer import build_all_indices, indices_exist
from config import INPUT_CSV, SAMPLE_CSV, OUTPUT_CSV, LOCAL_LLM

import pandas as pd
from dataclasses import asdict


AGENT_CSS = """
Screen {
    background: $surface;
}

#title-bar {
    dock: top;
    height: 3;
    background: #6c3ce9;
    color: white;
    text-align: center;
    padding: 1;
    text-style: bold;
}

#progress-section {
    height: 3;
    padding: 0 2;
}

#progress-label {
    width: 100%;
    text-align: center;
    color: $text-muted;
}

ProgressBar {
    padding: 0 2;
}

#ticket-panel {
    height: auto;
    max-height: 12;
    border: tall $primary;
    margin: 0 1;
    padding: 1;
}

#decision-panel {
    height: auto;
    max-height: 10;
    border: tall $success;
    margin: 0 1;
    padding: 1;
}

#stats-bar {
    dock: bottom;
    height: 3;
    background: $boost;
    padding: 1;
    text-align: center;
}

#log-panel {
    margin: 0 1;
    border: tall $accent;
    min-height: 8;
}

.panel-title {
    text-style: bold;
    color: $text;
}
"""


class TriageApp(App):
    """Support Triage Agent — Textual TUI."""

    CSS = AGENT_CSS
    TITLE = "🎯 Support Triage Agent (100% Local)"
    BINDINGS = [
        Binding("s", "start_processing", "Start", show=True),
        Binding("v", "validate_sample", "Validate", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self, input_csv=None, output_csv=None, **kwargs):
        super().__init__(**kwargs)
        self._input_csv = input_csv or INPUT_CSV
        self._output_csv = output_csv or OUTPUT_CSV
        self._results = []
        self._replied = 0
        self._escalated = 0
        self._processing = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            "🎯  Support Triage Agent  |  Press [bold]S[/] to start  |  [bold]V[/] to validate  |  [bold]Q[/] to quit",
            id="title-bar",
        )
        with Vertical():
            yield Static("Progress: waiting to start…", id="progress-label")
            yield ProgressBar(total=100, show_eta=True, id="progress")
            yield Static(
                "[bold]Current Ticket[/]\nWaiting to start…",
                id="ticket-panel",
            )
            yield Static(
                "[bold]Decision[/]\n—",
                id="decision-panel",
            )
            yield RichLog(highlight=True, markup=True, id="log-panel")
        yield Static(
            "✅ Replied: 0  |  ⚠️  Escalated: 0  |  📊 Total: 0",
            id="stats-bar",
        )
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#log-panel", RichLog)
        log.write(
            "[bold green]Agent ready.[/] Press S to process tickets or V to validate with sample data.\n"
            f"[dim]Using 100% Local RAG Pipeline: {LOCAL_LLM} (Reasoning) + FAISS (Retrieval).[/]"
        )

    # ── Actions ────────────────────────────────────────────────────────

    def action_start_processing(self) -> None:
        if self._processing:
            return
        self._processing = True
        self._results = []
        self._replied = 0
        self._escalated = 0
        self._run_pipeline(str(self._input_csv), str(self._output_csv))

    def action_validate_sample(self) -> None:
        if self._processing:
            return
        self._processing = True
        self._results = []
        self._replied = 0
        self._escalated = 0
        self._run_pipeline(str(SAMPLE_CSV), str(OUTPUT_CSV.parent / "sample_output.csv"))

    # ── Background worker ─────────────────────────────────────────────

    @work(thread=True)
    def _run_pipeline(self, input_csv: str, output_csv: str) -> None:
        worker = get_current_worker()
        log = self.query_one("#log-panel", RichLog)

        # Step 1: Ensure FAISS indices exist
        if not indices_exist():
            self.call_from_thread(log.write, "[bold yellow]Building FAISS indices (first run)…[/]")
            try:
                build_all_indices(
                    progress_cb=lambda msg: self.call_from_thread(log.write, f"  [dim]{msg}[/]")
                )
            except Exception as e:
                self.call_from_thread(log.write, f"[bold red]Index build error: {e}[/]")
                self._processing = False
                return

        # Step 2: Load tickets
        tickets = None
        try:
            import pandas as _pd
            df = _pd.read_csv(input_csv, keep_default_na=False)
            df.columns = [c.strip().lower() for c in df.columns]
            tickets = df.to_dict(orient="records")
        except Exception as e:
            self.call_from_thread(log.write, f"[bold red]CSV error: {e}[/]")
            self._processing = False
            return

        total = len(tickets)
        self.call_from_thread(self._set_progress_total, total)
        self.call_from_thread(log.write, f"\n[bold]Processing {total} tickets using {LOCAL_LLM}…[/]\n")

        results = []
        for i, ticket in enumerate(tickets):
            if worker.is_cancelled:
                break

            issue = ticket.get("issue", "")
            subject = ticket.get("subject", "")
            company = ticket.get("company", "")

            self.call_from_thread(
                self._update_ticket_panel,
                i + 1, total, company, subject, issue,
            )

            def _log(msg):
                self.call_from_thread(log.write, f"  {msg}")

            try:
                result = process_ticket(issue, subject, company, log_cb=_log)
                results.append(result)
                self.call_from_thread(self._update_decision_panel, result)
                self.call_from_thread(self._update_stats, result)
                self.call_from_thread(self._advance_progress, i + 1)
            except Exception as e:
                self.call_from_thread(log.write, f"[bold red]Error on ticket {i+1}: {e}[/]")
                results.append(TicketResult(
                    issue=issue, subject=subject, company=company,
                    status="escalated", request_type="product_issue",
                    product_area="unknown", response="Error processing ticket — escalated.",
                    justification=f"Processing error: {e}",
                ))
                self.call_from_thread(self._advance_progress, i + 1)

        # Step 3: Write output
        try:
            rows = [asdict(r) for r in results]
            out_df = _pd.DataFrame(rows)
            cols = ["issue", "subject", "company", "response", "product_area",
                    "status", "request_type", "justification"]
            out_df = out_df[cols]
            out_df.to_csv(output_csv, index=False)
            self.call_from_thread(
                log.write,
                f"\n[bold green]✅ Done! Results written to {output_csv}[/]",
            )
        except Exception as e:
            self.call_from_thread(log.write, f"[bold red]Write error: {e}[/]")

        self._processing = False

    # ── UI update helpers ─────────────────────────────────────────────

    def _set_progress_total(self, total: int) -> None:
        bar = self.query_one("#progress", ProgressBar)
        bar.update(total=total, progress=0)

    def _advance_progress(self, current: int) -> None:
        bar = self.query_one("#progress", ProgressBar)
        bar.update(progress=current)
        label = self.query_one("#progress-label", Static)
        label.update(f"Progress: {current}/{bar.total}")

    def _update_ticket_panel(
        self, idx: int, total: int, company: str, subject: str, issue: str,
    ) -> None:
        panel = self.query_one("#ticket-panel", Static)
        issue_preview = (issue[:200] + "…") if len(issue) > 200 else issue
        panel.update(
            f"[bold]Current Ticket ({idx}/{total})[/]\n"
            f"Company: [cyan]{company or 'None'}[/]  |  "
            f"Subject: [yellow]{subject or '—'}[/]\n"
            f"{issue_preview}"
        )

    def _update_decision_panel(self, result: TicketResult) -> None:
        panel = self.query_one("#decision-panel", Static)
        status_color = "green" if result.status == "replied" else "red"
        resp_preview = (result.response[:250] + "…") if len(result.response) > 250 else result.response
        panel.update(
            f"[bold]Decision[/]\n"
            f"Status: [{status_color}]{result.status.upper()}[/{status_color}]  |  "
            f"Type: [blue]{result.request_type}[/]  |  "
            f"Area: [magenta]{result.product_area}[/]\n"
            f"Response: {resp_preview}"
        )

    def _update_stats(self, result: TicketResult) -> None:
        if result.status == "replied":
            self._replied += 1
        else:
            self._escalated += 1
        total = self._replied + self._escalated
        bar = self.query_one("#stats-bar", Static)
        bar.update(
            f"✅ Replied: {self._replied}  |  "
            f"⚠️  Escalated: {self._escalated}  |  "
            f"📊 Total: {total}"
        )
