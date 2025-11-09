import src.configs.config
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.services.llm_chat_service import llm_chat_service

# 1. 创建一个提示模板
# 这个模板接收一个名为 "topic" 的输入变量
prompt = ChatPromptTemplate.from_template("给我讲一个关于 {topic} 的笑话")

# 2. 获取我们已经配置好的模型
# llm_chat_service.chat_model 内部就是我们自定义的 CustomGeminiChatModel
model = llm_chat_service.chat_model

# 3. 创建一个简单的输出解析器，将模型的聊天消息输出转换为字符串
output_parser = StrOutputParser()

# 4. 使用 LangChain 表达式语言 (LCEL) 将这三部分链接在一起
# 这就是“链式调用”
chain = prompt | model | output_parser

# 5. 调用链条，并传入输入变量
logger.info("正在调用链条...")
response = chain.invoke({"topic": "程序员"})

# 6. 打印结果
logger.info(f"链条返回的结果：\n{response}")
