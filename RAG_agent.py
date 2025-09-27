# from langchain_community.document_loaders import WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings


# urls = [
#     "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
#     "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
#     "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
# ]

# docs = [WebBaseLoader(url).load() for url in urls]


# print("docs:", docs[0][0].page_content.strip()[:1000])

# docs_list = [item for sublist in docs for item in sublist]

# print("docs_list:", docs_list)
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=100, chunk_overlap=50
# )
# doc_splits = text_splitter.split_documents(docs_list)


# print("doc_splits[0].page_content.strip()", doc_splits[0].page_content.strip())

# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vectorstore = InMemoryVectorStore.from_documents(
#     documents=doc_splits, embedding=embedding
# )
# retriever = vectorstore.as_retriever()

# from langchain.tools.retriever import create_retriever_tool

# retriever_tool = create_retriever_tool(
#     retriever,
#     "retrieve_blog_posts",
#     "Search and return information about Lilian Weng blog posts.",
# )

# retriever_tool.invoke({"query": "types of reward hacking"})



# from langgraph.graph import MessagesState
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# import os
# load_dotenv() 
# open_ai_kpi_key = os.getenv("open_ai_api_key")

# llm = ChatOpenAI(
#     # model="qwen-vl-plus",
#     model='qwen-plus-2025-09-11',
#     openai_api_key=open_ai_kpi_key,
#     openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     temperature=0.2,
#     timeout=30,
# )

# def generate_query_or_respond(state: MessagesState):
#     """Call the model to generate a response based on the current state. Given
#     the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
#     """
#     response = (
#         llm
#         .bind_tools([retriever_tool]).invoke(state["messages"])
#     )
#     return {"messages": [response]}

# #input = {"messages": [{"role": "user", "content": "hello!"}]}

# input = {
#     "messages": [
#         {
#             "role": "user",
#             "content": "What does Lilian Weng say about types of reward hacking?",
#         }
#     ]
# }
# generate_query_or_respond(input)["messages"][-1].pretty_print()



######################################################


from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()
open_ai_kpi_key = os.getenv("open_ai_api_key")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ===== 第一次运行：构建 & 保存向量库 =====
def build_and_save_vectorstore():
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = FAISS.from_documents(doc_splits, embedding)
    vectorstore.save_local("faiss_index")  # 保存到本地
    print("✅ 向量库已保存！")

#build_and_save_vectorstore()
# ===== 后续运行：直接加载向量库 =====
def load_vectorstore():
    vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    return vectorstore

# ===== 创建检索工具 =====
vectorstore = load_vectorstore()   # 如果是第一次运行，请先执行 build_and_save_vectorstore()
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)

# ===== LLM 配置 =====
llm = ChatOpenAI(
    model="qwen-plus-2025-09-11",
    openai_api_key=open_ai_kpi_key,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.2,
    timeout=30,
)

def generate_query_or_respond(state: MessagesState):
    """根据对话状态生成回复，决定是否调用检索工具"""
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


# 示例调用
input_msg = {
    "messages": [
        {"role": "user", "content": "What does Lilian Weng say about types of reward hacking?"}
    ]
}
generate_query_or_respond(input_msg)["messages"][-1].pretty_print()
