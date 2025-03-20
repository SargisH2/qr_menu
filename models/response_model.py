from pydantic import BaseModel

class OpenAIResponse(BaseModel):
    response: str
    tokens_used: int
