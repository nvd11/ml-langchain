import src.configs.config
from loguru import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.mcp import MCPTool
from src.services.llm_chat_service import llm_chat_service

# 1. 初始化一个指向公开 GitHub MCP 服务器的工具
# 这个工具会自动发现该服务器上所有可用的功能 (如 search_repositories, get_issue 等)
github_mcp_tool = MCPTool(
    mcp_server_name="github.com/modelcontextprotocol/servers/tree/main/src/github"
)

# 2. 将 MCP 工具放入我们的工具箱
# Agent 将能够使用这个工具箱里的所有工具
tools = [github_mcp_tool]

# 3. 从 LangChain Hub 下载一个标准的 Agent 提示模板
# 这个模板指导模型如何思考以及如何使用工具
prompt = hub.pull("hwchase17/react")

# 4. 获取我们配置好的模型
model = llm_chat_service.chat_model

# 5. 创建 Agent
# Agent 是驱动整个流程的核心，它会决定何时以及如何使用工具
agent = create_react_agent(model, tools, prompt)

# 6. 创建 Agent 执行器
# AgentExecutor 负责运行 Agent，并处理工具的调用和返回
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 7. 调用 Agent 执行器
# 我们提出一个问题，Agent 会分析它，并决定使用 github_mcp_tool 来回答
logger.info("正在调用 MCP Agent...")
response = agent_executor.invoke({"input": "帮我找一些关于 langchain 的 GitHub 仓库"})

logger.info(f"MCP Agent 返回的结果：\n{response}")
