############################



from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


from dotenv import load_dotenv
import os
import asyncio
load_dotenv() 

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

open_ai_kpi_key = os.getenv("open_ai_api_key")
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    # model="qwen-vl-plus",
    model='qwen-plus-2025-09-11',
    openai_api_key=open_ai_kpi_key,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.2,
    timeout=30,
)


# Set up MCP client
client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Make sure to update to the full absolute path to your math_server.py file
            "args": ["rh_mcp_server.py"],
            "transport": "stdio",
        }

    }
)

def main():

    tools = asyncio.run(client.get_tools())
    print("tools:", tools)
    
    
main()
