from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from groq import Groq
import os
from typing import Optional
import asyncio

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nuclear-reaction.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    user_message: str
    previous_user_message: Optional[str] = None
    previous_ai_message: Optional[str] = None
    chat_summary: Optional[str] = None

async def generate_stream(user_message: str, previous_user: Optional[str], previous_ai: Optional[str], summary: Optional[str]):
    """Generate streaming response from Groq"""
    try:
        # Build context
        messages = []
        
        # Add system message
        system_msg = "You are a helpful AI assistant explaining nuclear fission concepts. Keep responses concise and educational."
        if summary:
            system_msg += f"\n\nConversation summary so far: {summary}"
        
        messages.append({
            "role": "system",
            "content": system_msg
        })
        
        # Add previous exchange if available
        if previous_user and previous_ai:
            messages.append({"role": "user", "content": previous_user})
            messages.append({"role": "assistant", "content": previous_ai})
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Create streaming completion
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            stream=True
        )
        
        # Stream the response
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield f"data: {chunk.choices[0].delta.content}\n\n"
                await asyncio.sleep(0.01)  # Small delay for smooth streaming
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        print(f"Error in generate_stream: {str(e)}")
        yield f"data: [ERROR]\n\n"

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests with streaming response"""
    return StreamingResponse(
        generate_stream(
            request.user_message,
            request.previous_user_message,
            request.previous_ai_message,
            request.chat_summary
        ),
        media_type="text/event-stream"
    )

@app.get("/")
def get():
    return JSONResponse(content={"message": "Nuclear-Reaction AI API"}, status_code=200)

@app.head("/")
def ping():
    return JSONResponse(content={"message": "pong"}, status_code=200)
    
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
