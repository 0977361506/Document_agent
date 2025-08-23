import asyncio
from fastmcp import Client

client = Client("server.py")

async def call_tool():
    async with client:
        result = await client.call_tool("list_tools")
        print(result)

asyncio.run(call_tool())