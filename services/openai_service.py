from config import OPENAI_API_KEY
from prompts import PROMPT_DICT
from openai import OpenAI
from datetime import datetime
import json
from pydantic import BaseModel
from typing import List, Optional


class Recommendation(BaseModel):
    item: str
    price: str
    reason: str


class GPT_Message(BaseModel):
    response: str
    options: Optional[List[Recommendation]]


client = OpenAI(api_key=OPENAI_API_KEY)


class ChatBot:
    def __init__(self, connection, prompt_language="am"):
        self.connection = connection
        self.language = prompt_language
        self.system_message = {"role": "system", "content": PROMPT_DICT[prompt_language]}
        self.history = []

    async def ask(self, query):
        try:
            prompt_data = json.loads(query)
            current_time = prompt_data.get("time", datetime.now().strftime("%H:%M"))
            lang = prompt_data.get("language", "am")
            self.system_message = {"role": "system", "content": PROMPT_DICT[lang]}
            user_input = prompt_data.get("message", "")
        except json.JSONDecodeError:
            user_input = query
            current_time = datetime.now().strftime("%H:%M")

        message = {"role": "system", "content": f"Հիմա ժամը {current_time} է"}
        self.history.append(message)
        message = {"role": "user", "content": user_input}
        self.history.append(message)

        full_chat = [self.system_message] + self.history

        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=full_chat,
            response_format=GPT_Message,
        )
        recommendation = response.choices[0].message
        response_dict = json.loads(recommendation.model_dump()["content"])

        assistant_message = {
            "role": "assistant",
            "content": json.dumps(
                {
                    "response": response_dict["response"],
                    "options": response_dict["options"],
                }
            ),
        }
        self.history.append(assistant_message)

        await self.connection.send_json(self.history)
