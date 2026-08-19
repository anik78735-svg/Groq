import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq, RateLimitError, APIStatusError

app = FastAPI()

# Comma-separated keys ko list me load karein
RAW_KEYS = os.environ.get("GROQ_API_KEYS", "")
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

class CodeRequest(BaseModel):
    prompt: str

@app.post("/analyze-code")
async def analyze_code(request: CodeRequest):
    if not API_KEYS:
        raise HTTPException(status_code=500, detail="No Groq API keys configured.")

    last_error = None

    # Ek-ek karke saari keys try karega
    for idx, key in enumerate(API_KEYS):
        try:
            client = Groq(api_key=key)
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert senior software engineer. Analyze code, find bugs, optimize logic, and provide concise solutions."
                    },
                    {
                        "role": "user",
                        "content": request.prompt
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
            )
            
            return {
                "response": chat_completion.choices[0].message.content,
                "key_used_index": idx + 1
            }

        except (RateLimitError, APIStatusError) as e:
            # Agar rate limit (429) ya API error aaye to next key try karein
            last_error = str(e)
            continue
        except Exception as e:
            last_error = str(e)
            continue

    # Agar saari keys ki limit khatam ho chuki ho
    raise HTTPException(
        status_code=429,
        detail=f"All configured Groq API keys exhausted or failed. Last error: {last_error}"
    )
