"""Agno agents for generic competitor research."""

from datetime import date

from agno.agent import Agent
from agno.tools.tavily import TavilyTools

from config import get_gemini_model, load_company_name
from models import CompetitiveAnalysis

CURRENT_YEAR = date.today().year


def build_research_agent(company_profile: str) -> Agent:
    return Agent(
        name="ResearchAnalyst",
        model=get_gemini_model(),
        description="Competitive intelligence analyst with web search access.",
        tools=[TavilyTools(search_depth="advanced", format="markdown", include_answer=True)],
        instructions=[
            f"Today is {date.today().isoformat()}. The current year is {CURRENT_YEAR}.",
            "You produce competitor research using live web search.",
            "Answer ONLY what the user asked — do not add generic strategy essays.",
            f"When the question mentions 'this year', search for {CURRENT_YEAR} and late {CURRENT_YEAR - 1}.",
            "For social / TikTok / Instagram questions: find named campaigns, drops, hashtags, and creators.",
            "Include real source URLs for every claim.",
            "Do not invent campaigns. If nothing recent is found, say so.",
            "Use the company profile only to pick relevant competitors — do not repeat the profile back.",
            "",
            "Company profile:",
            company_profile,
        ],
        markdown=True,
    )


def build_analysis_agent(company_profile: str) -> Agent:
    return Agent(
        name="AnalysisWriter",
        model=get_gemini_model(),
        description="Turns research notes into a short, structured answer with a simple diagram.",
        instructions=[
            f"Today is {date.today().isoformat()}. The current year is {CURRENT_YEAR}.",
            "Convert research notes into CompetitiveAnalysis.",
            "direct_answer must directly answer the research question in 2-4 sentences.",
            "If the question is about campaigns, social, TikTok, or Instagram:",
            "  - Fill campaigns with one row per competitor + platform + campaign.",
            "  - Keep competitors list short: name, recent_moves (max 3 bullets), sources (URLs).",
            "  - Skip strengths/weaknesses/risk essays.",
            "key_findings: max 5 short bullets, only facts from search.",
            "opportunities_for_us: max 3 bullets, only if clearly supported.",
            "sources: top 5 URLs from the notes.",
            "mermaid_diagram rules:",
            "  - Use flowchart TB or flowchart LR only.",
            "  - Max 12 nodes total.",
            "  - One subgraph per competitor is fine.",
            "  - Label nodes like IG: Campaign name or TikTok: Campaign name.",
            "  - No mindmaps, no quadrant charts unless the question asks for positioning.",
            "",
            "Company profile:",
            company_profile,
        ],
        output_schema=CompetitiveAnalysis,
        use_json_mode=True,
        markdown=True,
    )


def run_competitive_research(company_profile: str, research_question: str) -> CompetitiveAnalysis:
    research_prompt = f"""
Research question:
{research_question}

Search the web and return research notes that answer this question only.
Focus on facts from {CURRENT_YEAR} (and late {CURRENT_YEAR - 1} if needed).
Include source URLs.
"""
    notes = build_research_agent(company_profile).run(research_prompt).content

    analysis_prompt = f"""
Research question:
{research_question}

Research notes:
{notes}

Return CompetitiveAnalysis.
Keep it short and question-focused.
Use a simple flowchart Mermaid diagram (max 12 nodes).
"""
    response = build_analysis_agent(company_profile).run(analysis_prompt)
    if not isinstance(response.content, CompetitiveAnalysis):
        raise ValueError(f"Expected CompetitiveAnalysis, got: {response.content}")
    return response.content
