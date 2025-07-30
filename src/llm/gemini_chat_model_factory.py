import src.configs.config
from src.configs.config import yaml_configs
from src.llm.llm_chat_model import LLMChatModelFactory
from typing import override
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiChatModelFactory(LLMChatModelFactory):
    def __init__(self, api_key=yaml_configs["gemini"]["api-key"]):
        self.api_key = api_key

 
    @override
    def build(self) -> BaseChatModel:

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            max_tokens=10000,
            api_key=self.api_key
        )
        return llm
    