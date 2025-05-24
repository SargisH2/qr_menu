from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from services.openai_service import ChatBot
import uvicorn
import uuid
import logging
import json
from prompts import prompt_rec_time, prompt_rec_orders
from typing import Dict, List
from contextlib import asynccontextmanager
from models import EntryLog, ButtonRequests, ChatHistory, Recommendation, GPT_Message
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sessions: Dict[str, ChatBot] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up")
    yield
    sessions.clear()
    logger.info("Application shutting down")

app = FastAPI(lifespan=lifespan)

app.mount("/admin_panel", StaticFiles(directory="admin_panel", html=True), name="admin")
app.mount("/build", StaticFiles(directory="build", html=True), name="main")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orders = {}


@app.get("/")
async def info():
    return "Welcome to the AI Chatbot API! version 07.05, count to 0 for simillar items: test: /admin_panel"

@app.get("/admin_panel", include_in_schema=False)
async def admin_index():
    return FileResponse("admin_panel/index.html")

@app.get("/build", include_in_schema=False)
async def main_page():
    return FileResponse("build/index.html")

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    user_id = str(uuid.uuid4())
    try:
        sessions[user_id] = ChatBot(websocket)
        chatbot = sessions[user_id]
        logger.info(f"New connection established. User ID: {user_id}")
        await websocket.send_text(f"Connected with session ID: {user_id}")
        while True:
            try:
                prompt = await websocket.receive_text()
                if not prompt.strip():
                    await websocket.send_text("Error: Empty message received")
                    continue
                logger.debug(f"Received message from {user_id}: {prompt}")
                await chatbot.ask(prompt, return_only_response = True)
            except ValueError as ve:
                logger.error(f"Validation error for {user_id}: {str(ve)}")
                await websocket.send_text(f"Error: Invalid input - {str(ve)}")
            except RuntimeError as re:
                logger.error(f"Runtime error for {user_id}: {str(re)}")
                await websocket.send_text(f"Error: Processing failed - {str(re)}")
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"Unexpected error for {user_id}: {str(e)}")
        await websocket.send_text(f"Error: Server error - {str(e)}")
    finally:
        if user_id in sessions:
            del sessions[user_id]
        logger.info(f"Session cleaned up for user {user_id}")
        try:
            await websocket.close()
        except Exception:
            pass

@app.post("/entry-log")
async def log_entry(entry: EntryLog):
    logger.info(f"Entry log received: {entry}")
    return {"status": "success", "received": entry}

@app.post("/button-requests")
async def log_button_requests(button_requests: ButtonRequests):
    logger.info(f"Button requests received: {button_requests.root}")
    return {"status": "success", "received": button_requests.root}

@app.post("/chat-history")
async def log_chat_history(chat_history: ChatHistory):
    logger.info(f"Chat history received: {chat_history.root}")
    return {"status": "success", "received": chat_history.root}

@app.get("/recommend/time", response_model=List[Recommendation])
async def recommend_by_time(language: str = "en"):
    language = language.lower()
    user_id = str(uuid.uuid4())
    chatbot = ChatBot(None, prompt_language=language)
    prompt = prompt_rec_time[language].format(current_time=datetime.now().strftime("%H:%M"))
    response = await chatbot.ask(prompt, return_only_response=True)
    return response.options or []

@app.post("/recommend/orders", response_model=GPT_Message)
async def recommend_by_orders(button_requests: ButtonRequests, language: str = "en"):
    language = language.lower()
    user_id = str(uuid.uuid4()) # test
    new_orders = set(req.id for req in button_requests.root)
    if not orders.get(user_id) or orders.get(user_id) != new_orders:
        orders[user_id] = {
            "orders": new_orders,
            "response": None
        }
    elif orders[user_id]["response"]:
        return orders[user_id]["response"]
    chatbot = ChatBot(None, prompt_language=language)
    order_summary = ", ".join([f"Button ID {req.id} at {req.timestamp}" for req in button_requests.root])
    prompt = prompt_rec_orders[language].format(orders=order_summary)
    response = await chatbot.ask(prompt, return_only_response=True)
    orders[user_id]['response'] = response
    return response

def run_server():
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            timeout_keep_alive=30
        )
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        raise

if __name__ == "__main__":
    run_server()