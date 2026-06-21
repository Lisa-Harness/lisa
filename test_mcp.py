import asyncio, os
from kernel.mcp_client import MCPMemoryClient
async def main():
    client = MCPMemoryClient('uv', ['run', '--directory', r'C:\Code\lisa_framework', '--package', 'mcp-server', 'python', '-m', 'mcp_server.server'])
    await client.store('query', 'response')
    print('success')
asyncio.run(main())
