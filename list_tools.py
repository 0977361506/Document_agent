import asyncio
from server import list_tools

async def main():
    """Display available Confluence MCP tools."""
    print("\nConfluence MCP Tools")
    print("==================")
    tools_result = await list_tools()
    
    for tool in tools_result['tools']:
        print(f"\n{tool['name']}")
        print("-" * len(tool['name']))
        print(f"{tool['description']}\n")
        
        if 'inputSchema' in tool:
            print("Input Schema:")
            properties = tool['inputSchema']['properties']
            required = tool['inputSchema'].get('required', [])
            
            max_name_len = max(len(name) for name in properties.keys())
            max_type_len = max(len(prop['type']) for prop in properties.values())
            
            for name, prop in properties.items():
                is_required = "(Required)" if name in required else "(Optional)"
                name_padding = " " * (max_name_len - len(name))
                type_padding = " " * (max_type_len - len(prop['type']))
                print(f"  {name}{name_padding} : {prop['type']}{type_padding} {is_required}")
                if prop.get('description'):
                    print(f"    {prop['description']}")
            print()

if __name__ == "__main__":
    asyncio.run(main()) 