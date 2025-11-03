import src.configs.config
from src.configs.config import yaml_configs
from src.llm.llm_chat_model import LLMChatModelFactory
from typing import override
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
import os


class GeminiChatModelFactory(LLMChatModelFactory):
    def __init__(self, api_key=yaml_configs["gemini"]["api-key"]):
        self.api_key = api_key

    @override
    def build(self) -> BaseChatModel:
        proxy_url = os.environ.get("https_proxy")
        client_options = None
        if proxy_url:
            client_options = {"api_endpoint": proxy_url}

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.2,
            max_tokens=None,
            google_api_key=self.api_key,
            client_options=client_options,
        )
        return llm
