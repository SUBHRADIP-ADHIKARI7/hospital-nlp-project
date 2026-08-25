"""
generate_more_data.py
----------------------
Use this once your 30-example starter dataset isn't enough (it won't be —
for a real system you'll eventually want a few hundred to a few thousand
labeled examples covering many phrasings, typos, languages, and edge cases).

This script asks Claude to generate NEW synthetic patient messages + labels,
in the same schema as data/synthetic_patient_data.json, and appends them.

Why synthetic data is fine to START with:
Real patient data is sensitive (see Layer 11 "Security & Compliance" in your
architecture — HIPAA/GDPR/NDHM). Synthetic data lets you build and test the
NLP pipeline without any compliance risk. Later, once the hospital has real
(properly consented and anonymized) patient messages, you'd mix those in
and re-validate — but that's a later-stage task, not day one.

HOW TO RUN:
    export ANTHROPIC_API_KEY="your-key-here"
    python generate_more_data.py --count 20 --category "elderly patients"
"""

import argparse
import json
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

GEN_PROMPT_TEMPLATE = """Generate {count} realistic, DIVERSE patient intake messages for a hospital
chat/voice/app system, focused on the category: "{category}".

Vary: sentence length, formality, typos, code-mixing with local languages if natural,
emergency vs non-emergency, and how clearly vs vaguely the patient describes things.

Return ONLY a JSON array, no markdown fences, where each item has this exact structure:
{{
  "text": "the patient message",
  "intent": one of ["emergency", "routine_consultation", "follow_up", "medication_refill", "information_request"],
  "symptoms": [list of strings],
  "body_part": string or null,
  "duration": string or null,
  "severity": one of ["mild", "moderate", "severe", "none"],
  "urgency": one of ["low", "medium", "high"],
  "suggested_department": string
}}
"""


def generate_batch(count: int, category: str) -> list:
    prompt = GEN_PROMPT_TEMPLATE.format(count=count, category=category)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def append_to_dataset(new_items: list, path: str = "../data/synthetic_patient_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    next_id = max(row["id"] for row in data) + 1
    for item in new_items:
        item["id"] = next_id
        next_id += 1
        data.append(item)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Added {len(new_items)} new examples. Dataset now has {len(data)} total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--category", type=str, default="general symptoms, mixed urgency")
    args = parser.parse_args()

    batch = generate_batch(args.count, args.category)
    append_to_dataset(batch)
