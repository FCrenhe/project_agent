import os
import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode

import asyncio
import uuid
from dotenv import load_dotenv


load_dotenv() 

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

open_ai_kpi_key = os.getenv("open_ai_api_key")
DB_URI =os.getenv("DB_URI")

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
from langchain_tavily import TavilySearch
from custom_tools import multiply, speak, weather

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


    
tools_mcp = asyncio.run(client.get_tools())
 

search_tool = TavilySearch(max_results=2)
multiply_tool = multiply 
speak_tool = speak
weather_tool = weather
tools = [search_tool, multiply_tool, speak_tool, weather_tool] + tools_mcp
graph_builder = StateGraph(State)
import os
#from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    # model="qwen-vl-plus",
    model='qwen-plus-2025-09-11',
    openai_api_key=open_ai_kpi_key,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.2,
    timeout=30,
)

llm_with_tools = llm.bind_tools(tools)
def chatbot(state: State):
    
    # messages = state["messages"]
    # response = await llm_with_tools.ainvoke(messages)
    # return {"messages": [response]}


    return {"messages": [llm_with_tools.invoke(state["messages"])]}
def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
tool_node = ToolNode(tools=tools)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")


graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")

#graph_builder.add_edge("chatbot", END)
store = InMemoryStore()

checkpointer = InMemorySaver()

graph = graph_builder.compile(store=store)

from IPython.display import Image, display
import pathlib

try:
    img_bytes = graph.get_graph().draw_mermaid_png()
    output_path = pathlib.Path("graph.png")
    with open(output_path, "wb") as f:
        f.write(img_bytes)
   # display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

async def stream_graph_updates(input_messages: list,
                         config: RunnableConfig):  
    print("ASSISTANT:", end="", flush=True)  

    async for chunk in graph.astream({"messages": input_messages}, 
                                      config, 
                                      stream_mode="messages"):
        if isinstance(chunk[0], AIMessageChunk):
            print(chunk[0].content, end="", flush=True)
        elif isinstance(chunk[0], ToolMessage):
            print(f"\nTOOL: {json.loads(chunk[0].content)}", flush=True)
            print("ASSISTANT:", end="", flush=True)  # 工具后继续追加模型内容
    print()

       

async def main():

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        with (
            PostgresStore.from_conn_string(DB_URI) as store,
    # PostgresSaver.from_conn_string(DB_URI) as checkpointer,
            ):
            config = {
                "configurable": {
                    "thread_id": "2",
                    "user_id": "1",
                }
            }
            
            if "remember" in user_input.lower():
                memory = "用户名字叫神人和"
                store.put(namespace, str(uuid.uuid4()), {"data": memory})
                
            user_id = config["configurable"]["user_id"]
            namespace = ("memories", user_id)
            memories = store.search(namespace, query=str(user_input))
            info = "\n".join([d.value["data"] for d in memories])
            system_msg = f"You are a helpful assistant talking to the user. User info: {info}"
        # last_message = user_input["messages"][-1]


            input_messages = [{"role": "system", "content": system_msg}] + [{"role": "user", "content": user_input}]
            
            await stream_graph_updates(input_messages, config)
        
asyncio.run(main())
