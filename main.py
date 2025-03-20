from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from services.openai_service import ChatBot
import uvicorn

app = FastAPI()

# CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}


@app.get("/")
async def get_status():
    return "running"


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Generate a unique user_id (for simplicity, using WebSocket object id; in production, use a proper UUID)
    user_id = id(websocket)

    if user_id not in sessions:
        sessions[user_id] = ChatBot(websocket)

    chatbot = sessions[user_id]

    while True:
        try:
            prompt = await websocket.receive_text()
            await chatbot.ask(prompt)
        except Exception as e:
            await websocket.send_text(f"Error: {str(e)}")
            if user_id in sessions:
                del sessions[user_id]
            break


if __name__ == "__main__":
    uvicorn.run(app)
