import src.configs.config
from src.configs.config import yaml_configs
from src.llm.llm_chat_model import LLMChatModelFactory
from typing import override
from langchain_core.language_models import BaseChatModel
import httpx
import os

# 一个简单的包装类，模仿 LangChain 的 BaseChatModel
class CustomGeminiChatModel(BaseChatModel):
    api_key: str
    model_name: str
    base_url: str

    def __init__(self, api_key: str, model_name: str, **kwargs):
        base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        super().__init__(api_key=api_key, model_name=model_name, base_url=base_url, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # httpx 会自动使用 https_proxy 环境变量
        client = httpx.Client(proxy=os.environ.get("https_proxy"))
        
        # 从 LangChain 的消息格式转换为 Gemini API 的格式
        prompt = messages[0].content
        
        json_data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        response = client.post(
            f"{self.base_url}?key={self.api_key}",
            json=json_data,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        
        # 从 Gemini API 的响应格式中提取文本
        text_response = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        
        # 返回 LangChain 期望的格式
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text_response))])

    @property
    def _llm_type(self) -> str:
        return "custom_gemini"

class GeminiChatModelFactory(LLMChatModelFactory):
    def __init__(self, api_key=yaml_configs["gemini"]["api-key"]):
        self.api_key = api_key

    @override
    def build(self) -> BaseChatModel:
        # 使用我们自己的包装类，而不是 LangChain 的类
        llm = CustomGeminiChatModel(
            api_key=self.api_key,
            model_name="gemini-2.0-flash"
        )
        return llm
