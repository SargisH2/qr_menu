from config import OPENAI_API_KEY
from prompts import PROMPT_DICT
from openai import OpenAI
from datetime import datetime
from models import GPT_Message
import json
from typing import List, Optional, Dict
import logging
import asyncio


client = OpenAI(api_key=OPENAI_API_KEY)
logger = logging.getLogger(__name__)

assistant_ids = {
    "am": "asst_wYkaLvIOu8yPQc48zott4udb",
    "en": "asst_wYkaLvIOu8yPQc48zott4udb",
    "ru": "asst_wYkaLvIOu8yPQc48zott4udb"
}

class ChatBot:
    def __init__(self, connection, prompt_language: str = "am"):
        self.connection = connection
        self.language = prompt_language
        self.assistant_id = assistant_ids[prompt_language]
        self.thread = client.beta.threads.create()
        self.user_message_times: List[str] = []
        self.history: List[Dict[str, str]] = []
        self.system_message: Dict[str, str] = {
            "role": "system",
            "content": PROMPT_DICT[prompt_language],
        }

    async def ask(self, query: str, return_only_response: bool = False) -> Optional[GPT_Message]:
        try:
            try:
                prompt_data: dict = json.loads(query)
                current_time: str = prompt_data.get("time", datetime.now().strftime("%H:%M"))
                lang: str = prompt_data.get("language", "am")
                user_input: str = prompt_data.get("message", "")
                if lang != self.language:
                    self.language = lang
                    self.system_message = {"role": "system", "content": PROMPT_DICT[lang]}
            except json.JSONDecodeError:
                user_input = query
                current_time = datetime.now().strftime("%H:%M")

            self.user_message_times.append(current_time)

            client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=user_input
            )

            system_message_with_time = f"{self.system_message['content']}\n\nCurrent time is {current_time}."
            run = client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant_id,
                instructions=system_message_with_time
            )

            while True:
                run = client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=run.id)
                if run.status == "completed":
                    break
                await asyncio.sleep(1)

            messages = client.beta.threads.messages.list(thread_id=self.thread.id, order="asc")
            history = []
            time_index = 0
            assistant_response = ""
            for msg in messages.data:
                if msg.role == "user":
                    time_message = {
                        "role": "system",
                        "content": PROMPT_DICT[self.language + "_time"].format(
                            current_time=self.user_message_times[time_index]
                        )
                    }
                    history.append(time_message)
                    time_index += 1
                elif msg.role == "assistant":
                    assistant_response = msg.content[0].text.value
                history.append({"role": msg.role, "content": msg.content[0].text.value})

            self.history = history


            try:
                response_data = json.loads(assistant_response)
                gpt_message = GPT_Message(**response_data)
            except (json.JSONDecodeError, ValueError):
                gpt_message = GPT_Message(response=assistant_response, options=None)

            if self.connection:
                await self.connection.send_json(self.history)
            if return_only_response:
                return gpt_message

        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            error_response = {"role": "assistant", "content": json.dumps({"error": str(e)})}
            if self.connection:
                await self.connection.send_json([error_response])
            if return_only_response:
                return GPT_Message(response=f"Error: {str(e)}", options=None)
