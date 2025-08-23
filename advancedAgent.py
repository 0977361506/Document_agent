import os
import re
import requests
from typing import TypedDict, List, Dict, Any
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
    urls_to_process: List[str]  # Queue of URLs to fetch
    processed_urls: List[str]  # URLs that have already been processed
    analysis_results: Dict[str, Any] # Structured analysis from LLM

class AnalysisResults(TypedDict):
    is_sufficient: bool
    summary: str

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

@tool("find_links")
def find_links(raw_html: str) -> List[str]:
    """
    Searches for Confluence page links (URLs or page IDs) within raw HTML content.
    
    Args:
        raw_html: The raw HTML content of a Confluence page.
        
    Returns:
        A list of unique Confluence page IDs found in the content.
    """
    # Regex to find /pages/viewpage.action?pageId=
    pattern = r'(?:\?pageId=|/display/[^/]+/)(?P<page_id>\d+)'
    
    found_ids = set()
    matches = re.finditer(pattern, raw_html)
    
    for match in matches:
        found_ids.add(match.group('page_id'))
        
    return list(found_ids)

# --- 3. Build the Graph (Xây dựng biểu đồ) ---
def build_confluence_agent_with_dict_router():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, tools=[get_confluence_page_by_id, find_links])

    # --- Define Nodes (Định nghĩa các nút) ---
    def fetch_page_node(state: AgentState) -> dict:
        print("-> Fetching pages from the queue...")
        urls_to_fetch = state["urls_to_process"]
        processed_content = ""
        current_urls = urls_to_fetch[:]
        state["urls_to_process"] = []
        
        for page_id in current_urls:
            if page_id not in state["processed_urls"]:
                print(f"  Fetching content for page ID: {page_id}")
                content = get_confluence_page_by_id.invoke({"page_id": page_id})
                processed_content += content
                state["processed_urls"].append(page_id)
        
        return {"raw_content": processed_content}

    def analyze_content_node(state: AgentState) -> dict:
        print("-> Analyzing content with LLM...")
        raw_content = state["raw_content"]
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một trợ lý thông minh chuyên phân tích tài liệu kỹ thuật. "
                     "Nhiệm vụ của bạn là trích xuất thông tin về các bảng dữ liệu liên quan đến chức năng, "
                     "và các đường link đến các tài liệu khác. Sau đó, đánh giá xem thông tin "
                     "về các bảng đã đủ để hoàn thành phân tích chưa. "
                     "Đưa ra kết quả dưới dạng JSON với 2 trường 'is_sufficient' (boolean) "
                     "và 'summary' (string)."),
            ("user", f"Đây là nội dung tài liệu: \n\n{raw_content}")
        ])
        
        chain = prompt_template | llm.with_structured_output(schema=AnalysisResults)
        response = chain.invoke({"raw_content": raw_content})

        new_links = find_links.invoke({"raw_html": raw_content})
        links_to_fetch = [link for link in new_links if link not in state["processed_urls"]]

        return {
            "analysis_results": response,
            "urls_to_process": links_to_fetch
        }
    
    # Nút này sẽ trả về một dictionary chứa tên nút tiếp theo
    def decide_node_dict(state: AgentState) -> dict:
        print("-> Deciding next step with dictionary...")
        analysis_results = state["analysis_results"]
        is_sufficient = analysis_results.get("is_sufficient")
        has_new_links = len(state["urls_to_process"]) > 0

        if is_sufficient or not has_new_links:
            print("-> Information is sufficient or no new links found. Ending.")
            return {"next_node": "final_summary"}
        else:
            print("-> Information is insufficient. Found new links. Continuing.")
            return {"next_node": "fetch_page"}
    
    def final_summary_node(state: AgentState) -> dict:
        print("-> Generating final summary...")
        final_summary = state["analysis_results"].get("summary")
        return {"final_output": final_summary}

    # --- Build the graph ---
    workflow = StateGraph(AgentState)
    
    workflow.add_node("fetch_page", fetch_page_node)
    workflow.add_node("analyze_content", analyze_content_node)
    # Thêm nút điều hướng mới
    workflow.add_node("decide_next", decide_node_dict)
    workflow.add_node("final_summary", final_summary_node)
    
    workflow.set_entry_point("fetch_page")

    workflow.add_edge("fetch_page", "analyze_content")
    workflow.add_edge("analyze_content", "decide_next")
    
    # Sử dụng lambda để trích xuất tên nút từ dictionary
    workflow.add_conditional_edges(
        "decide_next",
        lambda state: state["next_node"],
        {"fetch_page": "fetch_page", "final_summary": "final_summary"}
    )
    
    workflow.add_edge("final_summary", END)
    
    return workflow.compile()

# --- Main execution (Phần thực thi chính) ---
if __name__ == "__main__":
    app = build_confluence_agent_with_dict_router()
    page_id_to_fetch = "98379" 
    
    initial_state = {
        "user_input": page_id_to_fetch,
        "raw_content": "",
        "final_output": "",
        "urls_to_process": [page_id_to_fetch],
        "processed_urls": [],
        "analysis_results": {},
        "next_node": ""
    }
    
    print("Starting Confluence agent with dictionary router...")
    final_state = app.invoke(initial_state)
    
    print("\n--- Final Analysis (Phân tích cuối cùng) ---")
    print(final_state["final_output"])