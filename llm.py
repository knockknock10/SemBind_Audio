import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

response = client.chat.completions.create(
    model="openrouter/free",
    messages=[
        {
            "role": "user",
            "content": """
You are assisting with my SemBind-Audio research project.

Current pipeline:
Audio → STFT/Mel → HuBERT → semantic embedding
→ fingerprint → SHA-256.

I need to design and experimentally validate a method
for converting HuBERT representations into a temporal
semantic-importance map for selecting watermark embedding
regions.

Do NOT assume that attention magnitude equals semantic
importance. Propose experimentally defensible methods.

"""
        }
    ],
)

print("\n===== MODEL USED =====")
print(response.model)

print("\n===== RESPONSE =====")
print(response.choices[0].message.content)