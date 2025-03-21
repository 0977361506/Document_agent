import time
import asyncio
from server import (
    list_tools,
    confluence_search,
    confluence_get_spaces,
    confluence_get_page,
    confluence_create_page,
    confluence_update_page,
    confluence_delete_page
)

confluence_url = "https://arelis.atlassian.net/wiki"
access_token = "ATATT3xFfGF0ZnJDNUiibhs4IRWBWoBVAQXHeS-f5XAo493HD2o10DGLVosTNw5jbuAOHzdyU6mq6CIoFcOBe-CRynKMWtLmoZtlO6kGm9TAby2AZHD8g2HPDnD6KFuutA3369Dmzzc4936Ha82aUh0XLFhgr58bg5EbImndKcH4FaYfGVsO2NE=050B456E"

async def test_list_tools():
    print("\nTesting List Tools...")
    tools_result = await list_tools()
    print("\nAvailable MCP Tools")
    print("==================")
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

async def test_mcp_tools():
    print("\nTesting Confluence MCP Tools...")
    
    try:
        # Test 0: List Available Tools
        print("\n0. Testing List Tools...")
        await test_list_tools()
        
        # Test 1: Get Spaces
        print("\n1. Testing Get Spaces...")
        spaces_result = await confluence_get_spaces(
            confluence_url=confluence_url,
            access_token=access_token
        )
        
        if "isError" in spaces_result:
            print(f"Error getting spaces: {spaces_result['content'][0]['text']}")
        else:
            for content in spaces_result['content']:
                if content['type'] == 'text':
                    print(content['text'])
                elif content['type'] == 'json':
                    for space in content['data']['spaces']:
                        print(f"- {space['name']} ({space['key']})")
        
        # Test 2: Search Content
        print("\n2. Testing Search...")
        search_result = await confluence_search(
            confluence_url=confluence_url,
            access_token=access_token,
            query="Arelis Web Platform Architecture"
        )
        
        if "isError" in search_result:
            print(f"Error searching: {search_result['content'][0]['text']}")
        else:
            for content in search_result['content']:
                if content['type'] == 'text':
                    print(content['text'])
                elif content['type'] == 'json':
                    for result in content['data']['results']:
                        print(f"- {result['title']}")
        
        # Test 3: Page Operations
        print("\n3. Testing Page Operations...")
        space_key = "SD"
        test_page_title = f"Test Page {int(time.time())}"
        test_content = "<p>This is a test page created by MCP</p>"
        
        # Create page
        print(f"\nCreating page '{test_page_title}' in space '{space_key}'...")
        create_result = await confluence_create_page(
            confluence_url=confluence_url,
            access_token=access_token,
            space_key=space_key,
            title=test_page_title,
            content=test_content
        )
        
        if "isError" in create_result:
            print(f"Error creating page: {create_result['content'][0]['text']}")
            return
            
        for content in create_result['content']:
            if content['type'] == 'text':
                print(content['text'])
            elif content['type'] == 'json':
                page_id = content['data']['content_id']
                print(f"Page ID: {page_id}")
        
        # Read page
        print("\nReading created page...")
        read_result = await confluence_get_page(
            confluence_url=confluence_url,
            access_token=access_token,
            space_key=space_key,
            page_id=page_id
        )
        
        if "isError" in read_result:
            print(f"Error reading page: {read_result['content'][0]['text']}")
        else:
            for content in read_result['content']:
                if content['type'] == 'text':
                    print(content['text'])
                elif content['type'] == 'json':
                    print(f"Page title: {content['data']['title']}")
                    print(f"Page content: {content['data']['content']}")
                    print(f"Page version: {content['data']['version']}")
        
        # Update page
        print("\nUpdating page...")
        updated_content = "<p>This is a test page created by MCP - UPDATED</p>"
        update_result = await confluence_update_page(
            confluence_url=confluence_url,
            access_token=access_token,
            space_key=space_key,
            page_id=page_id,
            content=updated_content
        )
        
        if "isError" in update_result:
            print(f"Error updating page: {update_result['content'][0]['text']}")
        else:
            for content in update_result['content']:
                if content['type'] == 'text':
                    print(content['text'])
        
        # Delete page
        print("\nDeleting page...")
        delete_result = await confluence_delete_page(
            confluence_url=confluence_url,
            access_token=access_token,
            space_key=space_key,
            page_id=page_id
        )
        
        if "isError" in delete_result:
            print(f"Error deleting page: {delete_result['content'][0]['text']}")
        else:
            for content in delete_result['content']:
                if content['type'] == 'text':
                    print(content['text'])
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_mcp_tools()) 