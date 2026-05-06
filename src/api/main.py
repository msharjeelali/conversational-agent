import os
import sys
import uuid
import json
import torch
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datetime import datetime

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIALOG_PATH     = os.path.join(BASE_DIR,"..", "model/checkpoints/finetuned")
DIALOG_PATH     = os.path.abspath(DIALOG_PATH)
INTENT_PATH     = os.path.join(BASE_DIR, "..", "model/checkpoints/intent_classifier")
INTENT_PATH     = os.path.abspath(INTENT_PATH)
LOG_PATH        = os.path.join(BASE_DIR, "reports/conversation_logs.json")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

FAQ = {
    "what is your name":         "I am a conversational agent built with DialoGPT.",
    "who made you":              "I was built by Sharjeel and Mateen as a university project.",
    "what can you do":           "I can chat with you, answer questions, and hold a conversation.",
    "how are you":               "I am doing great, thanks for asking!",
    "what is the weather":       "I do not have access to live weather data, sorry!",
    "tell me a joke":            "Why did the computer go to the doctor? Because it had a virus!",
    "goodbye":                   "Goodbye! It was nice talking to you.",
    "bye":                       "Bye! Take care.",
    "hello":                     "Hello! How can I help you today?",
    "hi":                        "Hi there! What is on your mind?",
}

def faq_lookup(text: str):
    text = text.lower().strip().rstrip("?!.")
    for key, response in FAQ.items():
        if key in text:
            return response
    return None

print(f"Loading models on: {DEVICE}")
print("Loading DialoGPT...")
dialog_tokenizer = AutoTokenizer.from_pretrained(DIALOG_PATH)
dialog_model     = AutoModelForCausalLM.from_pretrained(
    DIALOG_PATH, dtype=torch.float32
).to(DEVICE)
dialog_model.eval()

print("Loading intent classifier...")
intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_PATH)
intent_model     = AutoModelForSequenceClassification.from_pretrained(
    INTENT_PATH
).to(DEVICE)
intent_model.eval()

ID_TO_INTENT = {0: "inform", 1: "question", 2: "directive", 3: "commissive"}

print("Models loaded successfully.")

sessions: dict[str, list[str]] = {}
MAX_HISTORY = 3

app = FastAPI(title="Conversational Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message:    str
    session_id: str = ""

class ChatResponse(BaseModel):
    response:   str
    intent:     str
    session_id: str

def predict_intent(text: str) -> str:
    enc = intent_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=64
    ).to(DEVICE)
    with torch.no_grad():
        logits = intent_model(**enc).logits
    return ID_TO_INTENT[logits.argmax(dim=-1).item()]

def generate_response(message: str, history: list[str]) -> str:
    context = ""
    for turn in history[-MAX_HISTORY:]:
        context += turn + dialog_tokenizer.eos_token
    context += message + dialog_tokenizer.eos_token

    inp_ids = dialog_tokenizer.encode(context)
    
    inp_ids = inp_ids[-256:]
    
    input_ids      = torch.tensor([inp_ids]).to(DEVICE)
    attention_mask = torch.ones_like(input_ids).to(DEVICE)

    with torch.no_grad():
        out = dialog_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=60,
            pad_token_id=dialog_tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.3
        )

    response = dialog_tokenizer.decode(
        out[:, input_ids.shape[1]:][0], skip_special_tokens=True
    ).strip()

    return response if response else "I am not sure how to respond to that."

def log_conversation(session_id: str, message: str, response: str, intent: str):
    log_entry = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "message":    message,
        "response":   response,
        "intent":     intent,
    }
    logs = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    logs.append(log_entry)
    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)

@app.get("/")
def root():
    return {"status": "running", "message": "Conversational Agent API is live"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = req.session_id if req.session_id else str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    message = req.message.strip().lower()

    try:
        loop   = asyncio.get_event_loop()
        intent = await asyncio.wait_for(
            loop.run_in_executor(None, predict_intent, message),
            timeout=10.0
        )

        response = None
        if intent == "question":
            response = faq_lookup(message)

        if not response:
            response = await asyncio.wait_for(
                loop.run_in_executor(None, generate_response, message, sessions[session_id]),
                timeout=30.0
            )

    except asyncio.TimeoutError:
        return ChatResponse(
            response="Sorry, I am taking too long to respond. Please try again.",
            intent="inform",
            session_id=session_id
        )

    sessions[session_id].append(message)
    sessions[session_id].append(response)
    log_conversation(session_id, message, response, intent)

    return ChatResponse(response=response, intent=intent, session_id=session_id)

@app.get("/logs")
def get_logs():
    if not os.path.exists(LOG_PATH):
        return {"logs": []}
    with open(LOG_PATH) as f:
        try:
            return {"logs": json.load(f)}
        except json.JSONDecodeError:
            return {"logs": []}

@app.get("/sessions")
def get_sessions():
    return {
        "active_sessions": len(sessions),
        "session_ids":     list(sessions.keys())
    }

@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    raise HTTPException(status_code=404, detail="Session not found")