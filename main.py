# Third-party imports
import openai
import os
from fastapi import FastAPI, Form

# Load env
from dotenv import load_dotenv
load_dotenv()

# Internal imports
from utils import send_message
from chatbot import cs_bot_response


app = FastAPI()
# Set up the OpenAI API client
openai.api_key = os.getenv("OPENAI_API_KEY")
whatsapp_number = os.getenv("TO_NUMBER")

print(cs_bot_response("超声波几钱?"))

@app.post("/message")
async def reply(Body: str = Form()):

    cs_bot_conversation = cs_bot_response(query=Body)
    send_message(whatsapp_number, cs_bot_conversation)
    return ""