import src.configs.config
from src.llm.gemini_chat_model_factory import GeminiChatModelFactory
from langchain_core.language_models import BaseChatModel


class LLMChatService:
    def __init__(self, chat_model: BaseChatModel | None = None):
        self.chat_model=chat_model
        if chat_model is None:
            self.chat_model = GeminiChatModelFactory().build()

    

    def query(self, question: str) -> str:
        return self.chat_model.invoke(question)

        
llm_chat_service = LLMChatService()