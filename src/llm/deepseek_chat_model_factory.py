import src.configs.config
from src.configs.config import yaml_configs
from src.llm.llm_chat_model import LLMChatModelFactory
from typing import override
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek

class DeepSeekChatModelFactory(LLMChatModelFactory):
    def __init__(self, 
                 api_key=yaml_configs["deepseek"]["api-key"],
                 base_url=yaml_configs["deepseek"]["base-url"]):
        self.api_key = api_key
        self.base_url = base_url

    @override
    def build(self) -> BaseChatModel:
        llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.2,
            max_tokens=None,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        return llm
