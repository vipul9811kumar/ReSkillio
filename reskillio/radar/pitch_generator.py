"""
reskillio/radar/pitch_generator.py
Gemini-powered outreach pitch generator for the Opportunity Radar.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from reskillio.radar.models import OpportunityMatch

logger = logging.getLogger(__name__)

PITCH_SYSTEM = (
    "You write concise, compelling outreach pitches for senior professionals "
    "approaching fractional or consulting opportunities. "
    "Tone: direct, confident, specific. No fluff. No generic phrases. "
    "Length: 3-5 sentences maximum. Every sentence earns its place. "
    "Ground everything in the candidate's actual experience."
)

PITCH_PROMPT = """\
Candidate background:
- Top skills: {skills}
- Years experience: {years}
- Scale managed: {scale}
- Professional identity: {identity}
- Key achievement: {achievement}

Opportunity:
- Company: {company} ({stage}, {industry})
- Role: {role}
- Type: {eng_type}
- Culture signals: {signals}
- Their key need: {key_need}

Write a 3-5 sentence outreach pitch. Be specific. Use the candidate's actual
scale and experience to address the company's specific need.
No subject line. Just the body paragraph.
"""

TIPS_PROMPT = """\
For a {eng_type} opportunity at a {stage} {industry} company,
provide 4 practical engagement tips as a JSON array:
[
  {{"label": "short label", "value": "specific actionable tip"}}
]
Tips should cover: timeline to first engagement, contract structure advice,
what to ask in first call, and one thing that typically kills these deals.
Return ONLY valid JSON. No markdown.
"""

INTRO_PROMPT = """\
Company: {company} in {industry}
Candidate background: {identity} with {years} years in {industry}

Suggest ONE warm intro angle in 1 sentence.
Example: "Check for Flexport alumni on LinkedIn — their COO came from there"
Be specific. No generic advice.
"""


def _apply_credentials() -> None:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    try:
        from config.settings import settings
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
    except Exception:
        pass


def _call_gemini(
    prompt: str,
    project_id: str,
    region: str,
    temperature: float = 0.65,
    max_tokens: int = 300,
    system: str = PITCH_SYSTEM,
) -> str:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

    from google import genai
    from google.genai import types as gt

    client = genai.Client(vertexai=True, project=project_id, location=region)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=gt.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=gt.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text.strip()


class PitchGenerator:
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region     = region
        _apply_credentials()

    def generate(
        self,
        match:              OpportunityMatch,
        candidate_profile:  dict,
    ) -> tuple[str, list[dict], Optional[str]]:
        """Returns (pitch_text, engagement_tips, intro_angle)."""
        opp         = match.opportunity
        skills      = ", ".join(candidate_profile.get("top_skills", [])[:5])
        identity    = candidate_profile.get("work_identity", "experienced professional")
        years       = candidate_profile.get("years_experience", "10+")
        scale       = candidate_profile.get("scale_signal", "large-scale operations")
        achievement = candidate_profile.get("key_achievement", "complex operational environments")
        signals     = ", ".join(opp.culture_signals[:3]) if opp.culture_signals else "growth stage"
        key_need    = opp.required_skills[0] if opp.required_skills else "operational leadership"

        pitch = self._generate_pitch(
            skills=skills, years=years, scale=scale, identity=identity,
            achievement=achievement, company=opp.company_name,
            stage=opp.company_stage.value, industry=opp.company_industry,
            role=opp.role_title, eng_type=opp.engagement_type.value,
            signals=signals, key_need=key_need,
        )
        tips  = self._generate_tips(
            eng_type=opp.engagement_type.value,
            stage=opp.company_stage.value,
            industry=opp.company_industry,
        )
        intro = self._generate_intro_angle(
            company=opp.company_name,
            industry=opp.company_industry,
            identity=identity,
            years=years,
        )
        return pitch, tips, intro

    def _generate_pitch(self, **kwargs) -> str:
        try:
            return _call_gemini(
                PITCH_PROMPT.format(**kwargs),
                self.project_id, self.region,
                temperature=0.65, max_tokens=300,
            )
        except Exception as e:
            logger.error(f"Pitch generation failed: {e}")
            return f"My experience in {kwargs.get('industry', 'operations')} at scale directly addresses what you need."

    def _generate_tips(self, **kwargs) -> list[dict]:
        try:
            text = _call_gemini(
                TIPS_PROMPT.format(**kwargs),
                self.project_id, self.region,
                temperature=0.2, max_tokens=250,
            )
            text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(text)[:4]
        except Exception as e:
            logger.error(f"Tips generation failed: {e}")
            return [
                {"label": "Timeline",    "value": "Expect 2-4 weeks from first call to signed agreement"},
                {"label": "Contract",    "value": "Start with a 90-day trial retainer before committing longer"},
                {"label": "First call",  "value": "Ask: what does success look like in 90 days?"},
                {"label": "Deal killer", "value": "Scope creep — define deliverables in writing before starting"},
            ]

    def _generate_intro_angle(self, **kwargs) -> Optional[str]:
        try:
            return _call_gemini(
                INTRO_PROMPT.format(**kwargs),
                self.project_id, self.region,
                temperature=0.5, max_tokens=80,
            )
        except Exception as e:
            logger.error(f"Intro angle failed: {e}")
            return None


_generator: Optional[PitchGenerator] = None

def get_pitch_generator(project_id: str, region: str = "us-central1") -> PitchGenerator:
    global _generator
    if _generator is None:
        _generator = PitchGenerator(project_id=project_id, region=region)
    return _generator
