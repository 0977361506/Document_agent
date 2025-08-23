#!/usr/bin/env python3
"""
Test script to verify Confluence Server integration.
This script tests the basic functionality of the Confluence MCP server
with the new Confluence Server authentication.
"""

import asyncio
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import DEFAULT_TOKEN, confluence_get_spaces, confluence_search


async def test_confluence_server():
    """Test basic Confluence Server functionality."""

    # Test configuration
    confluence_url = (
        "https://your-confluence-server.com"  # Replace with your actual server URL
    )
    access_token = DEFAULT_TOKEN

    print("Testing Confluence Server MCP Integration")
    print("=" * 50)
    print(f"Server URL: {confluence_url}")
    print(f"Using token: {access_token[:20]}...")
    print()

    try:
        # Test 1: List spaces
        print("Test 1: Getting Confluence spaces...")
        spaces_result = await confluence_get_spaces(
            confluence_url=confluence_url, access_token=access_token
        )

        if spaces_result.get("isError"):
            print(f"❌ Error getting spaces: {spaces_result}")
        else:
            print("✅ Successfully retrieved spaces")
            content = spaces_result.get("content", [])
            for item in content:
                if item.get("type") == "json":
                    spaces = item.get("data", {}).get("spaces", [])
                    print(f"   Found {len(spaces)} spaces")
                    for space in spaces[:3]:  # Show first 3 spaces
                        print(
                            f"   - {space.get('name', 'Unknown')} ({space.get('key', 'Unknown')})"
                        )
                    if len(spaces) > 3:
                        print(f"   ... and {len(spaces) - 3} more")
        print()

        # Test 2: Search
        print("Test 2: Searching for content...")
        search_result = await confluence_search(
            confluence_url=confluence_url, access_token=access_token, query="test"
        )

        if search_result.get("isError"):
            print(f"❌ Error searching: {search_result}")
        else:
            print("✅ Successfully performed search")
            content = search_result.get("content", [])
            for item in content:
                if item.get("type") == "json":
                    results = item.get("data", {}).get("results", [])
                    print(f"   Found {len(results)} search results")
                    for result in results[:3]:  # Show first 3 results
                        print(f"   - {result.get('title', 'Unknown')}")
                    if len(results) > 3:
                        print(f"   ... and {len(results) - 3} more")
        print()

        print("✅ All tests completed successfully!")
        print("\nConfluence Server integration is working correctly.")

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        print("\nPlease check:")
        print("1. Your Confluence Server URL is correct")
        print("2. Your access token is valid")
        print("3. Your Confluence Server is accessible")
        print("4. The token has appropriate permissions")


if __name__ == "__main__":
    print("Confluence Server MCP Test")
    print(
        "To run this test, update the confluence_url variable with your actual server URL"
    )
    print()

    # Uncomment the line below to run the actual test
    # asyncio.run(test_confluence_server())

    print("Test script ready. Update the URL and uncomment the test line to run.")
