import src.configs.config
from loguru import logger
from src.services.llm_chat_service import llm_chat_service

 
def test_query():
    logger.info("hello2")
    result = llm_chat_service.query("how sky is blue？")
    logger.info(result)
#Error discovering pytest tests(see Output> Python)