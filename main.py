import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq, RateLimitError, APIStatusError

app = FastAPI()

# Saari alag-alag keys ko ek list me automatically collect karega
API_KEYS = []
for i in range(1, 11):  # Key 1 se 10 tak search karega
    key = os.environ.get(f"GROQ_API_KEY_{i}")
    if key and key.strip():
        API_KEYS.append(key.strip())

# Backup check: agar single GROQ_API_KEY ho to wo bhi le lega
single_key = os.environ.get("GROQ_API_KEY")
if single_key and single_key.strip():
    API_KEYS.append(single_key.strip())

class CodeRequest(BaseModel):
    prompt: str

@app.post("/analyze-code")
async def analyze_code(request: CodeRequest):
    if not API_KEYS:
        raise HTTPException(
            status_code=500, 
            detail="No Groq API keys found. Set GROQ_API_KEY_1, GROQ_API_KEY_2 in Render."
        )

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
                "status": "success",
                "key_used": f"GROQ_API_KEY_{idx + 1}",
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
        detail=f"All keys exhausted. Last error: {last_error}"
    )
