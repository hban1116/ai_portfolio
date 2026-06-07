from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(prompt, label):
    print(f"\n{'='*50}")
    print(f"Technique: {label}")
    print(f"{'='*50}")
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    print(response.choices[0].message.content)

test_text = "The product broke after one day"

# Zero-shot
ask(
    f"Classify this as positive, negative or neutral: '{test_text}'",
    "Zero-shot"
)

# Few-shot
ask(
    f"""Classify sentiment. Examples:
'I love this' → positive
'This is broken' → negative
'It arrived' → neutral

Now classify: '{test_text}'""",
    "Few-shot"
)

# Chain of thought
ask(
    f"""Classify sentiment. Think step by step before answering.
Text: '{test_text}'""",
    "Chain of thought"
)

# Structured output
ask(
    f"""Classify sentiment. Reply ONLY in this JSON format, nothing else:
{{"sentiment": "positive/negative/neutral", "confidence": 0.0-1.0, "reason": "one sentence"}}

Text: '{test_text}'""",
    "Structured output"
)