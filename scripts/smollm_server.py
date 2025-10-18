#!/usr/bin/env python3
"""
Simple SmolLM3-3B server using transformers with OpenAI-compatible API.
"""

import time
from typing import Dict, List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.1
    max_tokens: int = 100
    stream: bool = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

class SmolLMServer:
    """Simple SmolLM3-3B server."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "HuggingFaceTB/SmolLM3-3B"
        
    def load_model(self):
        """Load the SmolLM3-3B model."""
        print(f"🚀 Loading {self.model_id} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device != "cpu" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print(f"✅ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def generate_response(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 100) -> str:
        """Generate response from messages."""
        if not self.model or not self.tokenizer:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Use Modal's exact prompt format
        try:
            # Build prompt exactly like Modal does
            prompt_parts = []
            
            # Process messages in Modal format
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    prompt_parts.insert(0, content)  # System prompt goes first
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            
            prompt_parts.append("Assistant:")  # Prompt for response
            prompt = "\n\n".join(prompt_parts)
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

# Global server instance
server = SmolLMServer()

# FastAPI app
app = FastAPI(title="SmolLM3-3B API Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    success = server.load_model()
    if not success:
        raise Exception("Failed to load SmolLM3-3B model")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": server.model_id,
        "device": server.device,
        "model_loaded": server.model is not None
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    
    # Convert Pydantic models to dicts
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    try:
        response_text = server.generate_response(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=server.model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting SmolLM3-3B API Server")
    print("📊 Server will run on http://localhost:8890")
    print("🤖 Compatible with OpenAI API format")
    
    uvicorn.run(app, host="0.0.0.0", port=8890)
