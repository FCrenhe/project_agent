from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
from langchain_openai import ChatOpenAI
import uuid

from dotenv import load_dotenv
load_dotenv()
import os


import debugpy

print("attach")
debugpy.listen(5678)
debugpy.wait_for_client()

open_ai_kpi_key = os.getenv("open_ai_api_key")

print("open_ai_kpi_key", open_ai_kpi_key)

llm = ChatOpenAI(
    # model="qwen-vl-plus",
    model='qwen-plus-2025-09-11',
    openai_api_key=open_ai_kpi_key,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.2,
    timeout=30,
)
DB_URI = "postgresql://postgres:123@localhost:5442/postgres?sslmode=disable"
with (
    PostgresStore.from_conn_string(DB_URI) as store,
   # PostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    # store.setup()
    # checkpointer.setup()
  #  print("a")
    
    def call_model(
        state: MessagesState,
        config: RunnableConfig,
        *,
        store: BaseStore,
    ):
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)
        memories = store.search(namespace, query=str(state["messages"][-1].content))
        
        
        info = "\n".join([d.value["data"] for d in memories])
        
        system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

        # Store new memories if the user asks the model to remember
        last_message = state["messages"][-1]
        if "remember" in last_message.content.lower():
            memory = "User name is renhe"
            store.put(namespace, str(uuid.uuid4()), {"data": memory})
            
        response = llm.invoke(
            [{"role": "system", "content": system_msg}] + state["messages"]
        )
        return {"messages": response}
        
        
    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")
    graph = builder.compile(
       # checkpointer=checkpointer,
        store=store,
    )
    # config = {
    #     "configurable": {
    #         "thread_id": "1",
    #         "user_id": "1",
    #     }
    # }
    # for chunk in graph.stream(
    #     {"messages": [{"role": "user", "content": "Hi! Remember: my name is renhe"}]},
    #     config,
    #     stream_mode="values",
    # ):
    #     chunk["messages"][-1].pretty_print()

    config = {
        "configurable": {
            "thread_id": "2",
            "user_id": "1",
        }
    }
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "what is my name?"}]},
        config,
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()
        
        
        
        
        
        
        
        