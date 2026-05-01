"""
Main entry point for the Support Triage Agent.

Usage:
    python main.py              # Launch Textual TUI (default)
    python main.py --batch      # Batch mode — process all tickets, no TUI
    python main.py --sample     # Validate against sample_support_tickets.csv
    python main.py --build-index# Only build the FAISS indices, then exit
    python main.py --limit N    # Process only N tickets (works with --batch and --sample)
    python main.py --ask "Q"    # Ask a specific question interactively
"""

import sys
import os

# Ensure the code/ directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
import urllib.request


def is_ollama_running():
    """Check if the local Ollama daemon is running."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def main():
    # Load environment variables from .env
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    from config import INPUT_CSV, SAMPLE_CSV, OUTPUT_CSV, LOCAL_LLM
    from indexer import build_all_indices, indices_exist

    # Verify Ollama is running
    if not is_ollama_running():
        print("❌ Ollama is not running on localhost:11434.")
        print("   Please start Ollama and ensure you have pulled the model:")
        print(f"   Run: ollama run {LOCAL_LLM}")
        sys.exit(1)

    args = sys.argv[1:]

    # ── Build index mode ──────────────────────────────────────────────
    if "--build-index" in args:
        print("Building local FAISS indices…")
        build_all_indices(progress_cb=lambda msg: print(f"  {msg}"))
        print("✅ Done.")
        return

    # ── Ask mode ──────────────────────────────────────────────────────
    if "--ask" in args:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--ask", type=str, required=True, help="The issue/question text")
        parser.add_argument("--company", type=str, default="", help="Company context")
        parser.add_argument("--subject", type=str, default="", help="Subject of the issue")
        parsed, _ = parser.parse_known_args(args)
        
        from agent import process_ticket
        
        if not indices_exist():
            print("FAISS indices not found. Building now…")
            build_all_indices(progress_cb=lambda msg: print(f"  {msg}"))
            
        print(f"\nProcessing Issue: '{parsed.ask}'")
        if parsed.company: print(f"Company: {parsed.company}")
        print(f"Using Local LLM: {LOCAL_LLM}")
        print("-" * 50)
        
        result = process_ticket(
            issue=parsed.ask,
            subject=parsed.subject,
            company=parsed.company,
            log_cb=lambda msg: print(f"  {msg}")
        )
        
        print("\n" + "=" * 50)
        print(f"Status:       {result.status.upper()}")
        print(f"Request Type: {result.request_type}")
        print(f"Product Area: {result.product_area}")
        print(f"\nJustification:\n{result.justification}")
        print(f"\nResponse:\n{result.response}")
        print("=" * 50)
        return

    # ── Batch mode ────────────────────────────────────────────────────
    if "--batch" in args or "--sample" in args:
        from agent import process_all_tickets
        
        limit = None
        if "--limit" in args:
            idx = args.index("--limit")
            if idx + 1 < len(args):
                limit = int(args[idx + 1])

        if "--sample" in args:
            csv_in = str(SAMPLE_CSV)
            csv_out = str(OUTPUT_CSV.parent / "sample_output.csv")
            print(f"🧪 Validating against sample tickets: {csv_in}")
        else:
            csv_in = str(INPUT_CSV)
            csv_out = str(OUTPUT_CSV)
            print(f"📋 Processing tickets: {csv_in}")

        if limit:
            print(f"⚠️ Limiting to {limit} tickets.")

        print(f"Using Local LLM: {LOCAL_LLM}")

        # Build index if missing
        if not indices_exist():
            print("FAISS indices not found. Building now…")
            build_all_indices(progress_cb=lambda msg: print(f"  {msg}"))

        # Process
        results = process_all_tickets(
            csv_path=csv_in,
            output_path=csv_out,
            limit=limit,
            log_cb=lambda msg: print(msg),
            progress_cb=lambda i, t, r: print(
                f"  [{i}/{t}] {r.status.upper()} — {r.request_type} — {r.product_area}"
            ),
        )

        # Summary
        replied = sum(1 for r in results if r.status == "replied")
        escalated = sum(1 for r in results if r.status == "escalated")
        print(f"\n{'='*50}")
        print(f"✅ Replied: {replied}  |  ⚠️  Escalated: {escalated}  |  Total: {len(results)}")
        print(f"📄 Output: {csv_out}")
        return

    # ── TUI mode (default) ────────────────────────────────────────────
    from tui import TriageApp

    app = TriageApp()
    app.run()


if __name__ == "__main__":
    main()
