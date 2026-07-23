"""Build a short markdown report with a simple Mermaid diagram."""

from __future__ import annotations

from models import Campaign, CompetitiveAnalysis


def _bullet_list(items: list[str], empty: str = "_None noted._") -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


def _campaign_table(campaigns: list[Campaign]) -> str:
    if not campaigns:
        return "_No named campaigns found in search results._"

    lines = [
        "| Competitor | Platform | Campaign | When | Detail |",
        "|---|---|---|---|---|",
    ]
    for row in campaigns:
        lines.append(
            f"| {row.competitor} | {row.platform} | {row.campaign} | {row.when} | {row.detail} |"
        )
    return "\n".join(lines)


def build_markdown_report(company_name: str, analysis: CompetitiveAnalysis) -> str:
    competitor_lines: list[str] = []
    for competitor in analysis.competitors:
        sources = ", ".join(competitor.sources) if competitor.sources else "_No URLs_"
        moves = (
            "\n".join(f"- {move}" for move in competitor.recent_moves)
            if competitor.recent_moves
            else "- _No moves._"
        )
        competitor_lines.append(f"**{competitor.name}**\n{moves}\nSources: {sources}")
    competitor_block = "\n\n".join(competitor_lines) if competitor_lines else "_No competitor notes._"

    opportunities_block = ""
    if analysis.opportunities_for_us:
        opportunities_block = f"""
## Quick opportunities for {company_name}
{_bullet_list(analysis.opportunities_for_us)}
"""

    sources_block = _bullet_list(analysis.sources, "_No URLs captured._")

    return f"""# Competitive Research Report

**Company:** {company_name}

## Your question
{analysis.research_question}

## Answer
{analysis.direct_answer}

## Campaign snapshot
{_campaign_table(analysis.campaigns)}

## Key findings
{_bullet_list(analysis.key_findings)}

## By competitor
{competitor_block}
{opportunities_block}
## Visual map

```mermaid
{analysis.mermaid_diagram.strip()}
```

## Sources
{sources_block}
"""
