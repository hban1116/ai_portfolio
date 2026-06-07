from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a precise data extraction engine.
Extract information from job listing text and return ONLY valid JSON.
If a field is missing, use null. Never guess. JSON only.
Do NOT wrap the JSON in markdown code blocks. Return raw JSON only."""

JOB_SCHEMA = {
    "company": "string or null",
    "role": "string",
    "salary_min": "number or null",
    "salary_max": "number or null",
    "location": "string or null",
    "remote": "boolean or null",
    "experience_years": "number or null",
    "skills": "array of strings or null"
}

JOB_LISTINGS = [
    "Google is hiring a Senior ML Engineer in NYC. Salary $180k-$220k. Must have 5+ years experience with Python, TensorFlow and Kubernetes. Hybrid role.",
    "Exciting opportunity at a fast growing startup! We need a React Developer who knows TypeScript and GraphQL. Remote OK. Competitive salary.",
    "Anthropic — AI Safety Researcher. San Francisco or remote. $200k-$300k. PhD preferred. Experience with LLMs, RLHF, and interpretability research required.",
    "Join our team as a Data Analyst. Excel and SQL skills needed. Office based in Chicago. 2-3 years experience. $70,000-$90,000.",
    "We are looking for a passionate Full Stack Engineer to help build the future of fintech."
]

def clean_json(raw):
    """Strip markdown code blocks if model wraps JSON in them."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = lines[1:-1]
        raw = "\n".join(lines)
    return raw.strip()

def extract_job(text, index):
    prompt = f"""Extract job data matching this schema exactly:
{json.dumps(JOB_SCHEMA, indent=2)}

Job listing: {text}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    cleaned = clean_json(raw)

    try:
        parsed = json.loads(cleaned)
        print(f"\n✅ Listing {index + 1}: {parsed.get('role')} at {parsed.get('company') or 'Unknown'}")
        print(json.dumps(parsed, indent=2))
        return parsed
    except json.JSONDecodeError:
        print(f"\n❌ Listing {index + 1} failed")
        print(f"Raw output was:\n{raw}")  # ← shows exactly what model returned
        return None

def main():
    print("🔍 Job Listing Extractor")
    print(f"Processing {len(JOB_LISTINGS)} listings...\n")

    results = []
    for i, listing in enumerate(JOB_LISTINGS):
        result = extract_job(listing, i)
        if result:
            results.append(result)

    with open("extracted_jobs.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Saved {len(results)} jobs to extracted_jobs.json")
    print(f"✅ Success rate: {len(results)}/{len(JOB_LISTINGS)}")

if __name__ == "__main__":
    main()