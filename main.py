from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class Input(BaseModel):
    text: str

@app.post("/ai")
def ai_response(data: Input):
    # مدل رایگان هوش مصنوعی از HuggingFace
    model_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"

    # بدون کلید هم کار می‌کند، فقط کندتر
    payload = {"inputs": data.text}
    response = requests.post(model_url, json=payload)

    return {"reply": response.json()}
