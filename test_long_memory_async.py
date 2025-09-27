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

# Initialize the model
#model = init_chat_model("anthropic:claude-3-5-sonnet-latest")

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

async def main():

    tools = await client.get_tools()
    # print("tools:", tools)
    # print("type_tools:", type(tools))
    
    for i in range(len(tools)):
        print(f"tools{i}:----------", tools[i])
    model_with_tools = model.bind_tools(tools)

    tool_node = ToolNode(tools)
    
    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

# Define call_model function
    async def call_model(state: MessagesState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}
    
    
    # Build the graph
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        should_continue,
    )
    builder.add_edge("tools", "call_model")

    # Compile the graph
    graph = builder.compile()

    math_response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )
    
    print("math_response:", math_response)

asyncio.run(main())
