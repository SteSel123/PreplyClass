"""Flexible Pydantic schemas for generic competitor research."""

from pydantic import BaseModel, Field


class Campaign(BaseModel):
    """One competitor campaign on a social or commerce platform."""

    competitor: str
    platform: str = Field(..., description="Instagram, TikTok Shop, TikTok, etc.")
    campaign: str = Field(..., description="Campaign or initiative name")
    when: str = Field(..., description="Timing, e.g. Q1 2026 or March 2026")
    detail: str = Field(..., description="One short line on what they are doing")


class Metric(BaseModel):
    label: str
    value: str


class CompetitorInsight(BaseModel):
    name: str
    recent_moves: list[str] = Field(default_factory=list, description="Only facts tied to the question")
    sources: list[str] = Field(default_factory=list, description="Real URLs from web search")


class CompetitiveAnalysis(BaseModel):
    """Structured research output — keeps answers tight and question-focused."""

    research_question: str
    direct_answer: str = Field(
        ...,
        description="2-4 sentences that directly answer the research question.",
    )
    campaigns: list[Campaign] = Field(
        default_factory=list,
        description="Campaign rows when the question is about marketing, social, or product launches.",
    )
    key_findings: list[str] = Field(default_factory=list, max_length=5)
    competitors: list[CompetitorInsight] = Field(default_factory=list)
    opportunities_for_us: list[str] = Field(default_factory=list, max_length=3)
    sources: list[str] = Field(default_factory=list, description="Top source URLs used")
    mermaid_diagram: str = Field(
        ...,
        description=(
            "Simple valid Mermaid diagram without fences. "
            "Prefer flowchart TB or LR with at most 12 nodes. No mindmaps."
        ),
    )
