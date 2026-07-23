"""
Competitive Research Platform

1. Set your company knowledge base in company_profile.md
2. Ask any research question
3. Get structured JSON + a markdown report with a Mermaid diagram

Usage:
  python main.py "Who are our top competitors and how are they priced?"
  python main.py "What product launches did rival skincare D2C brands make this year?"

Requires GEMINI_API_KEY and TAVILY_API_KEY in .env
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

from agents import run_competitive_research
from config import REPORT_PATH, RESEARCH_JSON_PATH, configure_gemini_env, load_company_name, load_company_profile
from report import build_markdown_report

load_dotenv()
configure_gemini_env()


def require_api_keys() -> None:
    missing = [key for key in ("GEMINI_API_KEY", "TAVILY_API_KEY") if not os.getenv(key)]
    if missing:
        print(f"Missing API keys: {', '.join(missing)}")
        print("Add them to .env (see .env.example).")
        sys.exit(1)


def require_question() -> str:
    if len(sys.argv) < 2:
        print('Usage: python main.py "Your research question"')
        print('Example: python main.py "Map our top 4 competitors and their pricing"')
        sys.exit(1)
    return " ".join(sys.argv[1:])


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def save_outputs(company_name: str, analysis) -> None:
    REPORT_PATH.parent.mkdir(exist_ok=True)
    RESEARCH_JSON_PATH.write_text(json.dumps(analysis.model_dump(), indent=2), encoding="utf-8")
    REPORT_PATH.write_text(build_markdown_report(company_name, analysis), encoding="utf-8")


def main() -> None:
    require_api_keys()
    question = require_question()
    company_profile = load_company_profile()
    company_name = load_company_name(company_profile)

    header("Competitive Research Platform")
    print("\nLoaded company profile from company_profile.md")
    print(f"\nResearch question:\n{question}")

    header("Phase 1 — Web research")
    print("Searching the web with Tavily...")

    header("Phase 2 — Structured analysis + Mermaid diagram")
    analysis = run_competitive_research(company_profile, question)
    save_outputs(company_name, analysis)

    header("Deliverables")
    print(f"\nReport:  {REPORT_PATH}")
    print(f"JSON:    {RESEARCH_JSON_PATH}")
    print(f"\nSummary: {analysis.direct_answer}")


if __name__ == "__main__":
    main()
