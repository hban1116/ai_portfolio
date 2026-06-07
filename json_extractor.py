from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a precise data extraction engine.
Extract information from text and return ONLY valid JSON.
If a field is missing from the text, use null.
Never guess or hallucinate values.
No explanation. No markdown. JSON only."""

def extract(text, schema, label=""):
    prompt = f"""Extract data from this text and return JSON matching exactly this schema:
{json.dumps(schema, indent=2)}

Text: {text}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        print(f"\n✅ {label}")
        print(json.dumps(parsed, indent=2))
        return parsed
    except json.JSONDecodeError:
        print(f"\n❌ {label} — Invalid JSON returned:")
        print(raw)
        return None

# ---- Schema definitions ----
job_schema = {
    "company": "string",
    "role": "string",
    "salary": "string or null",
    "location": "string or null",
    "remote": "boolean or null"
}

review_schema = {
    "product": "string",
    "sentiment": "positive/negative/neutral",
    "rating": "number 1-5 or null",
    "main_complaint": "string or null",
    "would_recommend": "boolean or null"
}

person_schema = {
    "name": "string",
    "age": "number or null",
    "occupation": "string or null",
    "location": "string or null"
}

# ---- Test extractions ----
extract(
    "Amazon is hiring a Remote Senior Data Engineer in Seattle, paying $160k-$200k per year.",
    job_schema,
    "Job listing"
)

extract(
    "Terrible product. Paid $80 for this headphone and it stopped working in 2 weeks. Would not recommend to anyone. 1 star.",
    review_schema,
    "Product review"
)

extract(
    "My name is Priya Sharma, I'm a 28 year old software developer based in Bangalore.",
    person_schema,
    "Person info"
)

extract(
    "We are looking for a passionate Frontend Developer to join our team.",
    job_schema,
    "Incomplete job listing"
)