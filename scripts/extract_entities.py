"""
extract_entities.py
--------------------
This script is the "NLP & Intent Understanding" + "Information Extraction"
component from the architecture diagram (Layer 1: Input Processing).

WHAT IT DOES:
Takes raw patient text (typed, or transcribed from voice) and turns it into
structured data the rest of the system (Triage Agent, Specialist Agent,
Availability Agent) can use.

Input example:
    "I'm having severe chest pain and can't breathe, started 10 minutes ago"

Output example:
    {
      "intent": "emergency",
      "symptoms": ["chest pain", "difficulty breathing"],
      "body_part": "chest",
      "duration": "10 minutes",
      "severity": "severe",
      "urgency": "high",
      "suggested_department": "Emergency / Cardiology"
    }

HOW IT WORKS:
Instead of training a custom NLP model from scratch (which needs lots of data,
GPU compute and ML expertise), this uses a large language model (Claude) as
the "understanding engine". We give it a strict instruction (a "prompt") that
tells it exactly what fields to extract and in what format. This is the
standard, practical approach used in most real-world agentic systems today,
and it's exactly what layer 9 ("NLP / LLM Models") in your diagram refers to.

HOW TO RUN:
1. Install the dependency:
       pip install anthropic
2. Set your API key as an environment variable:
       export ANTHROPIC_API_KEY="your-key-here"
3. Run:
       python extract_entities.py
"""

import json
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# This is the extraction schema — the fields your whole downstream system
# (Triage Agent, Routing Engine) will rely on. Keep this in sync with
# data/synthetic_patient_data.json so your training/eval labels match
# exactly what production extracts.
SYSTEM_PROMPT = """You are a medical intake NLP system for a hospital patient routing platform.

Given a patient's free-text message (which may be informal, contain typos, be code-mixed
with local languages, or be vague), extract the following fields and return ONLY valid JSON,
no other text, no markdown fences:

{
  "intent": one of ["emergency", "routine_consultation", "follow_up", "medication_refill", "information_request"],
  "symptoms": [list of symptom strings, empty list if none],
  "body_part": string or null,
  "duration": string or null (how long the issue has been present, in the patient's own words),
  "severity": one of ["mild", "moderate", "severe", "none"],
  "urgency": one of ["low", "medium", "high"],
  "suggested_department": string (best-guess hospital department)
}

Rules:
- If there is any mention of inability to breathe, unconsciousness, heavy bleeding,
  stroke-like symptoms, or severe trauma, intent="emergency" and urgency="high".
- If unsure about a field, make the best reasonable medical judgment call rather than
  leaving it blank, but never invent symptoms the patient did not mention or imply.
- Keep symptom names short and clinical (e.g. "chest pain", not "my chest kind of hurts").
"""


def extract(patient_text: str) -> dict:
    """Send patient text to the model and get back structured JSON."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": patient_text}],
    )

    raw_text = response.content[0].text.strip()
    # Defensive cleanup in case the model wraps output in ```json fences
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"error": "Could not parse model output", "raw_output": raw_text}


def run_on_dataset(path: str, limit: int = 5):
    """Quick sanity check: run extraction on a few rows from your labeled dataset
    and compare the model's output to your ground-truth labels."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for row in data[:limit]:
        print("=" * 60)
        print("INPUT TEXT :", row["text"])
        print("EXPECTED   :", {k: row[k] for k in ["intent", "urgency", "suggested_department"]})
        result = extract(row["text"])
        print("MODEL SAID :", result)


if __name__ == "__main__":
    # Example single call
    sample = "I'm having severe chest pain and can't breathe, started 10 minutes ago"
    print("Single example run:")
    print(json.dumps(extract(sample), indent=2))

    print("\nBatch run against labeled dataset (first 5 rows):")
    run_on_dataset("../data/synthetic_patient_data.json", limit=5)
