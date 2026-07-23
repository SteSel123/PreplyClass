# Competitive Research Platform

A generic **Agno + Tavily + Gemini** agent that researches competitors for **your** company.

## How it works

1. **Set your company context** in `company_profile.md` (your knowledge base)
2. **Ask any research question** on the command line
3. **Get deliverables:**
   - `output/competitive_report.md` — markdown report with a **Mermaid diagram**
   - `output/research.json` — structured Pydantic output

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add to `.env`:
- `GEMINI_API_KEY`
- `TAVILY_API_KEY`

## Run

```bash
python main.py "Who are our top 4 competitors and how are they priced?"
python main.py "What social commerce moves are rival D2C brands making this year?"
python main.py "Compare AI coding tools targeting data science teams"
```

## Project layout

```
company_profile.md   Your company knowledge base — edit this first
main.py              Entry point
agents.py            Research + analysis agents
models.py            Flexible Pydantic schemas
report.py            Builds the markdown report
config.py            Paths and Gemini setup
output/
  competitive_report.md
  research.json
```

## Teaching story

> "Every company needs competitive intelligence. You keep your company profile in one
> file. When leadership asks a question, the agent searches the web, structures the
> answer, and writes a report with a diagram — tailored to **your** business."

Model: `gemini-2.5-flash`
