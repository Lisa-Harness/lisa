import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

async def main():
    params = StdioServerParameters(
        command=r"C:\Code\lisa_framework\.venv\Scripts\lisa-memory.exe",
        args=["--data-dir", r"C:\Code\lisa_framework\hosts\little_lisa\artifacts"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool("add_memory", {"user_slug": "test", "content": "smoke"})
            res = await session.call_tool("search_memory", {"user_slug": "test", "query": "smoke"})
            print(res.content[0].text if res.content else "no content")
asyncio.run(main())
