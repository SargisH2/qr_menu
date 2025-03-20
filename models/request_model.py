from pydantic import BaseModel

class OpenAIRequest(BaseModel):
    prompt: str
