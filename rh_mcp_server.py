
from mcp.server.fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool()
def add(a: int, b: int) -> int:
    print("add")
    return a + b




if __name__ == "__main__":
    print("run")
   # mcp.run(transport="stdio")
    
    mcp.run(transport="tcp", host="127.0.0.1", port=12345)
