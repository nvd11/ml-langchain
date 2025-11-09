from src.llm.deepseek_chat_model_factory import DeepSeekChatModelFactory
from loguru import logger

def test_deepseek_chat_model_invoke():
    """
    Tests the invocation of the DeepSeek chat model.
    """
    # 1. Create an instance of the model factory
    factory = DeepSeekChatModelFactory()

    # 2. Build the chat model
    model = factory.build()

    # 3. Invoke the model with a simple prompt
    prompt = "Hello, who are you?"
    result = model.invoke(prompt)

    # 4. Log the response and assert that it's valid
    logger.info(f"DeepSeek model response: {result.content}")
    assert result.content is not None
    assert len(result.content) > 0
