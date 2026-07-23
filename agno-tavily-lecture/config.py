"""Paths and Gemini model setup."""

import os
from pathlib import Path

from agno.models.google import Gemini

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"

COMPANY_PROFILE_PATH = ROOT / "company_profile.md"
REPORT_PATH = OUTPUT_DIR / "competitive_report.md"
RESEARCH_JSON_PATH = OUTPUT_DIR / "research.json"

GEMINI_MODEL = "gemini-2.5-flash"


def configure_gemini_env() -> None:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key


def get_gemini_model() -> Gemini:
    configure_gemini_env()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return Gemini(id=GEMINI_MODEL, api_key=api_key)


def load_company_profile() -> str:
    if not COMPANY_PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {COMPANY_PROFILE_PATH.name}. "
            "Create your company knowledge base before running research."
        )
    return COMPANY_PROFILE_PATH.read_text(encoding="utf-8").strip()


def load_company_name(profile: str | None = None) -> str:
    text = profile or load_company_profile()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "## Company name" and i + 1 < len(lines):
            return lines[i + 1].strip()
    return "Your company"
