from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio
client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Replace with absolute path to your math_server.py file
            "args": ["D:\MIT\code\agent\agent_project\rh_mcp_server.py"],
            "transport": "stdio",
        },
        "weather": {
            # Ensure you start your weather server on port 8000
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    }
)

async def main():
    tools = await client.get_tools()
    # agent = create_react_agent(
    #     "anthropic:claude-3-7-sonnet-latest",
    #     tools
    # )
    # math_response = await agent.ainvoke(
    #     {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    # )
    # weather_response = await agent.ainvoke(
    #     {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
    # )
    
asyncio.run(main())
