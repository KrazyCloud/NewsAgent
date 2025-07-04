import json
import logging
import re
import ast
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, ValidationError
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
import uvicorn

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("Claim_Expert_Agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- Initialize LLM ----------
llm = OllamaLLM(model="mistral")

hypothesis_expert_prompt = """
You are a multilingual reasoning expert AI, capable of evaluating whether a result supports a claim.

You must handle:
- Inputs in any Indian language (Hindi, Tamil, Bengali, etc.)
- Romanized or transliterated forms (e.g., "ye sach hai", "modi ji ne bola tha")
- Code-mixed expressions (e.g., Hindi-English)
- Claims and results written in **different scripts or languages**
- References, links, or citations included in the result

Your goal:
- Carefully analyze the **semantic relationship** between the "claim" and "result"
- Extract factual and logical evidence from the result (including links or references)
- Determine if the result **supports**, **contradicts**, or is **neutral** toward the claim

When providing your answer:
- Always return **valid JSON**
- Include a detailed explanation (with contextual translation if needed)
- Highlight how any included **links or citations** support or contradict the claim

Respond in **this JSON format**:
{{
  "supports_claim": true or false,
  "explanation": "A detailed explanation mentioning if and how the result supports the claim, citing key evidence, translation if necessary, and use of any provided links."
}}
"""

# ---------- Prompt Chain ----------
hypothesis_prompt_template = ChatPromptTemplate.from_messages([
    ("system", hypothesis_expert_prompt),
    ("human", "Claim: {claim}\nResult: {result}")
])
hypothesis_chain: Runnable = hypothesis_prompt_template | llm

# ---------- Models ----------
class HypothesisCheckRequest(BaseModel):
    claim: str
    result: str

class HypothesisCheckResponse(BaseModel):
    supports_claim: bool
    explanation: str

# ---------- Utility Functions ----------
def clean_text(text: str) -> str:
    text = re.sub(r"[^\x00-\x7F\u0900-\u0D7F]+", " ", text)  # Keep ASCII + Indian scripts
    text = re.sub(r"\s+", " ", text).strip()
    return text

def truncate_tokens(text: str, max_tokens: int = 3000) -> str:
    approx_char_limit = max_tokens * 4  # 4 chars per token
    return text[:approx_char_limit]

# ---------- FastAPI App ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up Hypothesis Expert Agents...")
    try:
        hypothesis_chain.invoke({"claim": "Sample claim", "result": "Sample result"})
        logger.info("Warm-up successful.")
    except Exception as e:
        logger.error("Warm-up failed: %s", str(e))
    yield

app = FastAPI(title="Dual Expert API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Main Endpoint ----------
@app.post("/agents/hypothesis-check", response_model=HypothesisCheckResponse)
async def check_hypothesis(payload: HypothesisCheckRequest):
    try:
        logger.info("Received Claim and Result")

        # Clean and truncate
        cleaned_claim = truncate_tokens(clean_text(payload.claim))
        cleaned_result = truncate_tokens(clean_text(payload.result))

        logger.info(f"Claim: {cleaned_claim}")
        logger.info(f"Result: {cleaned_result[:500]}... [truncated]")

        # LLM Call
        response = await hypothesis_chain.ainvoke({
            "claim": cleaned_claim,
            "result": cleaned_result
        })

        response_text = str(response).strip()
        logger.info(f"Raw LLM Response: {response_text}")

        # Parse response safely
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(response_text)
            except Exception as e:
                logger.error("Failed to parse LLM response as JSON or literal: %s", str(e))
                raise HTTPException(status_code=500, detail="Error: LLM response was not valid JSON or dict.")

        # Handle case where model returns only "message"
        if "message" in parsed and ("supports_claim" not in parsed or "explanation" not in parsed):
            logger.warning("LLM returned only 'message'. Substituting default values for missing fields...")
            parsed = {
                "supports_claim": False,
                "explanation": (
                    f"Model did not respond with expected fields. Fallback explanation used:\n{parsed['message']}"
                )
            }


        # Validate against Pydantic schema
        try:
            validated = HypothesisCheckResponse(**parsed)
        except ValidationError as ve:
            logger.error(f"Validation error: {ve}")
            raise HTTPException(
                status_code=500,
                detail="LLM response did not contain required fields (supports_claim, explanation)."
            )

        return validated

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        logger.error(f"Unexpected error during hypothesis check: {e}")
        raise HTTPException(status_code=500, detail="Unexpected internal error.")


# ---------- Health Check ----------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ---------- Logging Middleware ----------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5005, reload=False)
