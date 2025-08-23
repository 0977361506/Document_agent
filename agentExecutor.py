import os
import re
import requests
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field

# Load environment variables
load_dotenv()

# --- 1. Define Tools (Công cụ) ---
@tool
def get_confluence_page_by_id(page_id: str) -> str:
    """
    Fetches the content of a Confluence page by its ID.

    Args:
        page_id: The unique ID of the Confluence page.

    Returns:
        The raw HTML content of the page as a string.
        Returns an error message if the request fails.
    """
    confluence_url = os.getenv("CONFLUENCE_URL")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")

    if not confluence_url or not api_token:
        return "Error: Confluence URL or API token is not set in environment variables."
    
    api_endpoint = f"{confluence_url}/rest/api/content/{page_id}?expand=body.storage"
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_token}"
    }

    try:
        response = requests.get(api_endpoint, headers=headers)
        response.raise_for_status()
        data = response.json()
        raw_content = data['body']['storage']['value']
        return raw_content
    except requests.exceptions.RequestException as e:
        return f"Error fetching page from Confluence: {e}"

@tool("find_links")
def find_links(raw_html: str) -> List[str]:
    """
    Searches for Confluence page links (URLs or page IDs) within raw HTML content.
    
    Args:
        raw_html: The raw HTML content of a Confluence page.
        
    Returns:
        A list of unique Confluence page IDs found in the content.
    """
    pattern = r'(?:\?pageId=|/display/[^/]+/)(?P<page_id>\d+)'
    
    found_ids = set()
    matches = re.finditer(pattern, raw_html)
    
    for match in matches:
        found_ids.add(match.group('page_id'))
        
    return list(found_ids)

# --- 2. Build the Agent (Xây dựng Agent) ---
def create_confluence_agent():
    # Initialize the LLM with a Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    tools = [get_confluence_page_by_id, find_links]

    # Create the agent prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Bạn là một trợ lý thông minh chuyên phân tích tài liệu Confluence. "
                       "Nhiệm vụ của bạn là lấy nội dung trang, phân tích thông tin về các bảng dữ liệu, luồng chức năng và các đường link liên quan. "
                       "Nếu bạn cần thêm thông tin để có một phân tích đầy đủ, bạn phải tự động tìm kiếm các link trong tài liệu và truy cập các trang đó. "
                       "Hãy trả lời bằng tiếng Việt."),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create a tool-calling agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# --- 3. Main execution (Phần thực thi chính) ---
if __name__ == "__main__":
    agent = create_confluence_agent()
    
    # Yêu cầu của người dùng ban đầu
    user_input = "Phân tích nội dung và các bảng chức năng liên quan từ trang Confluence có ID là 98379."
    
    print("Starting Confluence agent...")
    try:
        response = agent.invoke({"input": user_input})
        print("\n--- Phân tích cuối cùng (Final Analysis) ---")
        print(response["output"])
    except Exception as e:
        print(f"An error occurred: {e}")