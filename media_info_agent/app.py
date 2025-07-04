import json
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
import uvicorn
import re
import ast

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("Media_Info_Extractor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- Initialize LLM ----------
llm = OllamaLLM(model="mistral:7b")

# ---------- Prompt: Media Info Extractor ----------
media_info_prompt = """
You are a data extractor. Extract the following fields from this unstructured post into JSON:
- title
- duration
- likes
- views
- publisher
- published_date

Respond only with valid JSON.

Here is the input:
{{your_unstructured_text}}
"""

# ---------- Create Chain ----------
media_info_prompt_template = ChatPromptTemplate.from_messages([
    ("system", media_info_prompt),
    ("human", "Extract this post: {input_text}")
])
media_info_chain: Runnable = media_info_prompt_template | llm

# ---------- Models ----------
class MediaInfoRequest(BaseModel):
    input_text: str

from typing import Optional

class MediaInfoResponse(BaseModel):
    title: Optional[str]
    duration: Optional[str]
    likes: Optional[str]
    views: Optional[str]
    publisher: Optional[str]
    published_date: Optional[str]


# ---------- FastAPI App ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up Media Info Expert Agent...")
    try:
        media_info_chain.invoke({"input_text": "Remembering S. P. Balasubrahmanyam on his Birth Anniversary."})
        logger.info("Warm-up successful.")
    except Exception as e:
        logger.error("Warm-up failed: %s", str(e))
    yield

app = FastAPI(title="Media Info Extractor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_text(text: str) -> str:
    text = re.sub(r"[^\x00-\x7F\u0900-\u0D7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- Endpoint: Media Info Extraction ----------
@app.post("/agents/media-info-extract", response_model=MediaInfoResponse)
async def extract_media_info(payload: MediaInfoRequest):
    try:
        logger.info("Received media info extraction request.")
        cleaned_text = clean_text(payload.input_text)
        logger.info(f"Input: {cleaned_text}")
        
        response = await media_info_chain.ainvoke({"input_text": cleaned_text})
        response_text = str(response).strip()
        
        # Robust JSON parsing
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(response_text)  # Fallback to handle single quotes etc.
        
        return parsed
    except Exception as e:
        logger.error(f"Media info extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Media info extraction failed.")

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Middleware: Logging ----------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# ---------- Run App ----------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5006, reload=False)
