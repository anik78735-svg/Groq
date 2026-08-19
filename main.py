import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq, RateLimitError, APIStatusError

app = FastAPI(
    title="Groq AI Code Assistant",
    description="Multi-key fallback API wrapper for Groq models",
    version="1.0.0"
)

# Render Environment Variables se keys collect karna
API_KEYS = []
for i in range(1, 11):
    key = os.environ.get(f"GROQ_API_KEY_{i}")
    if key and key.strip():
        API_KEYS.append(key.strip())

single_key = os.environ.get("GROQ_API_KEY")
if single_key and single_key.strip() and single_key.strip() not in API_KEYS:
    API_KEYS.append(single_key.strip())

class CodeRequest(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Groq API Backend is running. Visit /docs to test endpoints.",
        "configured_keys_count": len(API_KEYS)
    }

@app.post("/analyze-code")
async def analyze_code(request: CodeRequest):
    if not API_KEYS:
        raise HTTPException(
            status_code=500,
            detail="No Groq API keys found. Set GROQ_API_KEY_1, GROQ_API_KEY_2 in Render Environment variables."
        )

    last_error = None

    # Multi-key rotation loop
    for idx, key in enumerate(API_KEYS):
        try:
            client = Groq(api_key=key)

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert senior software engineer. "
                            "Analyze code, debug issues, provide optimizations, "
                            "and write clean, production-ready code solutions."
                        )
                    },
                    {
                        "role": "user",
                        "content": request.prompt
                    }
                ],
                # Active and supported model on Groq:
                model="llama-3.1-8b-instant",
                temperature=0.2,
                max_tokens=4096
            )

            return {
                "status": "success",
                "key_used": f"GROQ_API_KEY_{idx + 1}",
                "model": "llama-3.1-8b-instant",
                "response": chat_completion.choices[0].message.content
            }

        except (RateLimitError, APIStatusError) as e:
            last_error = str(e)
            continue
        except Exception as e:
            last_error = str(e)
            continue

    raise HTTPException(
        status_code=429,
        detail=f"All configured API keys exhausted or returned errors. Last error: {last_error}"
    )
