import os
from typing import Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="AI Terminal Builder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompletionRequest(BaseModel):
    provider: str = Field(..., description="Name of the provider: openai, openrouter, anthropic, google, mistral, groq, openai-compatible")
    model: str = Field(..., description="Model identifier for the selected provider")
    prompt: str = Field(..., description="Prompt or user message to send")
    system: Optional[str] = Field(None, description="Optional system instruction")
    api_key: Optional[str] = Field(None, description="API key/token for provider")
    # For OpenAI-compatible custom endpoints (Azure OpenAI, local servers, etc.)
    base_url: Optional[str] = Field(None, description="Custom base URL for OpenAI-compatible providers")
    extra_headers: Optional[Dict[str, str]] = Field(default=None, description="Any additional headers to include")
    temperature: Optional[float] = Field(default=0.2, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=512, ge=1)


class CompletionResponse(BaseModel):
    provider: str
    model: str
    output: str
    raw: Optional[Dict[str, Any]] = None


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


@app.get("/ai/providers")
def list_providers():
    return {
        "providers": [
            {
                "key": "openai",
                "name": "OpenAI",
                "notes": "Standard Chat Completions API",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo"]
            },
            {
                "key": "openrouter",
                "name": "OpenRouter",
                "notes": "Multi-provider hub via OpenAI-compatible API",
                "models": ["openrouter/auto", "anthropic/claude-3.5-sonnet", "google/gemini-1.5-pro", "meta-llama/llama-3.1-70b-instruct"]
            },
            {
                "key": "anthropic",
                "name": "Anthropic",
                "notes": "Claude Messages API",
                "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
            },
            {
                "key": "google",
                "name": "Google Gemini",
                "notes": "Generative Language API",
                "models": ["gemini-1.5-pro", "gemini-1.5-flash"]
            },
            {
                "key": "mistral",
                "name": "Mistral",
                "notes": "Mistral AI API (OpenAI-compatible)",
                "models": ["mistral-large-latest", "mistral-small-latest", "codestral-latest"]
            },
            {
                "key": "groq",
                "name": "Groq",
                "notes": "Fast Llama/Mixtral (OpenAI-compatible)",
                "models": ["llama-3.1-70b-versatile", "mixtral-8x7b"]
            },
            {
                "key": "openai-compatible",
                "name": "OpenAI-Compatible",
                "notes": "Bring any base_url + key (Azure OpenAI, local, vLLM, etc.)",
                "models": ["custom"]
            }
        ]
    }


def _extract_text_from_openai_like(resp_json: Dict[str, Any]) -> str:
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        return str(resp_json)


@app.post("/ai/complete")
def ai_complete(req: CompletionRequest):
    provider = req.provider.lower()

    if provider in ("openai", "openrouter", "mistral", "groq") or provider == "openai-compatible":
        base_url = req.base_url
        headers: Dict[str, str] = req.extra_headers.copy() if req.extra_headers else {}
        if provider == "openai":
            base_url = base_url or "https://api.openai.com/v1"
            if not req.api_key:
                raise HTTPException(status_code=400, detail="api_key required for OpenAI")
            headers.update({"Authorization": f"Bearer {req.api_key}"})
        elif provider == "openrouter":
            base_url = base_url or "https://openrouter.ai/api/v1"
            if not req.api_key:
                raise HTTPException(status_code=400, detail="api_key required for OpenRouter")
            headers.update({"Authorization": f"Bearer {req.api_key}", "HTTP-Referer": "https://app", "X-Title": "AI Terminal Builder"})
        elif provider == "mistral":
            base_url = base_url or "https://api.mistral.ai/v1"
            if not req.api_key:
                raise HTTPException(status_code=400, detail="api_key required for Mistral")
            headers.update({"Authorization": f"Bearer {req.api_key}"})
        elif provider == "groq":
            base_url = base_url or "https://api.groq.com/openai/v1"
            if not req.api_key:
                raise HTTPException(status_code=400, detail="api_key required for Groq")
            headers.update({"Authorization": f"Bearer {req.api_key}"})
        else:  # openai-compatible custom
            if not base_url:
                raise HTTPException(status_code=400, detail="base_url required for openai-compatible")
            if req.api_key:
                headers.update({"Authorization": f"Bearer {req.api_key}"})

        url = f"{base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": req.model,
            "messages": ([{"role": "system", "content": req.system}] if req.system else []) + [
                {"role": "user", "content": req.prompt}
            ],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": False,
        }
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json", **headers}, timeout=60)
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        data = response.json()
        text = _extract_text_from_openai_like(data)
        return {"provider": provider, "model": req.model, "output": text, "raw": data}

    if provider == "anthropic":
        if not req.api_key:
            raise HTTPException(status_code=400, detail="api_key required for Anthropic")
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": req.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": req.model,
            "max_tokens": req.max_tokens or 300,
            "temperature": req.temperature or 0.2,
            "messages": ([{"role": "user", "content": req.prompt}] if not req.system else [
                {"role": "user", "content": f"System: {req.system}\n\nUser: {req.prompt}"}
            ]),
            "system": req.system or None,
        }
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        data = response.json()
        try:
            content_blocks = data.get("content", [])
            text = "".join([b.get("text", "") for b in content_blocks if b.get("type") == "text"]) or str(data)
        except Exception:
            text = str(data)
        return {"provider": provider, "model": req.model, "output": text, "raw": data}

    if provider == "google":
        # Google Gemini - API key in query string
        if not req.api_key:
            raise HTTPException(status_code=400, detail="api_key required for Google Gemini")
        model = req.model
        base = req.base_url or "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base}/models/{model}:generateContent?key={req.api_key}"
        payload = {
            "contents": [{
                "role": "user",
                "parts": (([{"text": req.system}] if req.system else []) + [{"text": req.prompt}])
            }],
            "generationConfig": {
                "temperature": req.temperature,
                "maxOutputTokens": req.max_tokens
            }
        }
        response = requests.post(url, json=payload, headers={"content-type": "application/json"}, timeout=60)
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        data = response.json()
        try:
            candidates = data.get("candidates", [])
            text = candidates[0]["content"]["parts"][0].get("text", "") if candidates else str(data)
        except Exception:
            text = str(data)
        return {"provider": provider, "model": req.model, "output": text, "raw": data}

    raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
