import os
import requests
from typing import TypedDict, List
from dotenv import load_dotenv

# Import the Gemini model from the Google GenAI library
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# --- 1. Define State (Trạng thái) ---
class AgentState(TypedDict):
    """
    Represents the state of our agent.
    This object is passed between nodes and stores all necessary data.
    """
    user_input: str  # The initial input from the user (Confluence page ID)
    raw_content: str  # Raw content retrieved from the Confluence API
    final_output: str  # The final, formatted output for the user

# --- 2. Define Tools (Công cụ) ---
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
    # Get API credentials from environment variables
    confluence_url = os.getenv("CONFLUENCE_URL")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")

    if not confluence_url or not api_token:
        return "Error: Confluence URL or API token is not set in environment variables."
    
    # Construct the API endpoint URL
    api_endpoint = f"{confluence_url}/rest/api/content/{page_id}?expand=body.storage"
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_token}"
    }

    try:
        response = requests.get(api_endpoint, headers=headers)
        response.raise_for_status()  # Throws an HTTPError for bad responses (4xx or 5xx)
        
        data = response.json()
        raw_content = data['body']['storage']['value']
        return raw_content
    except requests.exceptions.RequestException as e:
        return f"Error fetching page from Confluence: {e}"

# --- 3. Build the Graph (Xây dựng biểu đồ) ---
def build_confluence_agent():
    # Initialize the LLM with a Gemini model
    # You must have a GOOGLE_API_KEY environment variable set up.
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # --- Define Nodes (Định nghĩa các nút) ---
    def fetch_page_node(state: AgentState) -> dict:
        """Node 1: Fetches the raw page content using the Confluence tool."""
        print("-> Fetching Confluence page...")
        page_id = state["user_input"]
        raw_content = get_confluence_page_by_id.invoke({"page_id": page_id})
        return {"raw_content": raw_content}

    def format_response_node(state: AgentState) -> dict:
        """Node 2: Formats the raw content into a user-friendly summary."""
        print("-> Formatting the response with LLM...")
        raw_content = state["raw_content"]
        
        # Check if the content is an error message
        if "Error:" in raw_content:
            return {"final_output": raw_content}

        # Prompt the LLM to format the content
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một trợ lý thông minh chuyên phân tích tài liệu kỹ thuật. "
                       "Hãy tóm tắt nội dung sau, tập trung vào các bảng dữ liệu liên quan, "
                       "luồng hoạt động của chức năng và các thông tin thiết kế chính. "
                       "Đảm bảo câu trả lời rõ ràng và dễ hiểu."),
            ("user", f"Đây là nội dung tài liệu: \n\n{raw_content}")
        ])
        
        chain = prompt_template | llm
        response = chain.invoke({"raw_content": raw_content})
        
        return {"final_output": response.content}

    # --- Build the graph ---
    workflow = StateGraph(AgentState)
    
    workflow.add_node("fetch_page", fetch_page_node)
    workflow.add_node("format_response", format_response_node)
    
    # Set entry point and define edges
    workflow.set_entry_point("fetch_page")
    workflow.add_edge("fetch_page", "format_response")
    workflow.add_edge("format_response", END)
    
    return workflow.compile()

# --- Main execution (Phần thực thi chính) ---
if __name__ == "__main__":
    app = build_confluence_agent()
    
    # Replace 'your_page_id' with a real Confluence page ID
    page_id_to_fetch = "98379" 
    
    initial_state = {
        "user_input": page_id_to_fetch,
        "raw_content": "",
        "final_output": ""
    }
    
    final_state = app.invoke(initial_state)
    
    print("\n--- Final Analysis (Phân tích cuối cùng) ---")
    print(final_state["final_output"])
