from __future__ import annotations

DEFAULT_TOPIC = (
    "We study the fusion of social media data and remote sensing data for "
    "flood monitoring and disaster response."
)

DEFAULT_RUBRIC: dict[int, str] = {
    100: "Perfect fit: directly fuses social media and remote sensing for flood events",
    90: "Flood plus either social media or remote sensing; methods highly relevant",
    80: "Flood-related and uses social media / remote sensing / crowdsourced data",
    70: "Flood-hazard remote sensing or social media, but data fusion is not the focus",
    60: "Social media / remote sensing fusion for other natural hazards (transferable)",
    50: "General disaster management or general remote sensing; flood not the focus",
    40: "Only marginally mentions floods or social media data",
    30: "Hydrology / meteorology, unrelated to disaster response",
    20: "Only an incidental keyword match",
    10: "Completely unrelated",
}


def build_rubric_text(rubric: dict[int, str]) -> str:
    return "\n".join(
        f"- {score}: {meaning}" for score, meaning in sorted(rubric.items(), reverse=True)
    )
