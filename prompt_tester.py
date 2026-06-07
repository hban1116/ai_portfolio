from groq import Groq
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def test_prompt(system_prompt, user_prompt, label="Test"):
    print(f"\n{'='*50}")
    print(f"Test: {label}")
    print(f"{'='*50}")

    start = time.time()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    elapsed = round(time.time() - start, 2)
    result = response.choices[0].message.content
    tokens = response.usage.total_tokens

    print(f"Response: {result}")
    print(f"Tokens used: {tokens} | Time: {elapsed}s")
    return result

# ---- Test 1: Vague system prompt ----
test_prompt(
    system_prompt="You are a helpful assistant.",
    user_prompt="Extract the company name, role and salary from this: 'Google is hiring a Senior ML Engineer at 180k'",
    label="Vague system prompt"
)

# ---- Test 2: Precise system prompt ----
test_prompt(
    system_prompt="""You are a data extraction engine.
Always reply ONLY with valid JSON in this exact format:
{"company": "string", "role": "string", "salary": "string"}
No explanation. No extra text. JSON only.""",
    user_prompt="Extract from this: 'Google is hiring a Senior ML Engineer at 180k'",
    label="Precise system prompt"
)

# ---- Test 3: Handle missing data ----
test_prompt(
    system_prompt="""You are a data extraction engine.
Always reply ONLY with valid JSON in this exact format:
{"company": "string", "role": "string", "salary": "string or null"}
No explanation. No extra text. JSON only.""",
    user_prompt="Extract from this: 'Meta is looking for a Data Scientist'",
    label="Missing data handling"
)