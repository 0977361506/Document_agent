import asyncio
import json
import os
import re
from typing import Any, Dict, List, TypedDict

import requests
import uvicorn
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv()

# Giới hạn số trang tối đa agent sẽ truy cập để tránh vòng lặp vô hạn
MAX_PAGES = 5


# --- 1. Define State (Trạng thái) ---
class AgentState(TypedDict):
    user_input: str
    raw_content: str
    final_output: str
    urls_to_process: List[str]
    processed_urls: List[str]
    analysis_results: Dict[str, Any]


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
    """
    confluence_url = os.getenv("CONFLUENCE_URL")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    if not confluence_url or not api_token:
        return "Error: Confluence URL or API token is not set in environment variables."
    api_endpoint = f"{confluence_url}/rest/api/content/{page_id}?expand=body.storage"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_token}"}
    try:
        response = requests.get(api_endpoint, headers=headers)
        response.raise_for_status()
        data = response.json()
        raw_content = data["body"]["storage"]["value"]
        return raw_content
    except requests.exceptions.RequestException as e:
        return f"Error fetching page from Confluence: {e}"


@tool("find_links")
def find_links(raw_html: str) -> List[str]:
    """
    Searches for Confluence page links (URLs or page IDs) within raw HTML content,
    prioritizing <a> tags and then scanning the rest of the text.
    Returns:
        A list of unique Confluence page IDs found in the content.
    """
    found_ids = set()

    # Define the regex pattern for a Confluence page ID
    page_id_pattern = r"(?:\?pageId=|/display/[^/]+/)(?P<page_id>\d+)"

    # --- Step 1: Parse HTML with BeautifulSoup to find links in <a> tags -
    # --
    soup = BeautifulSoup(raw_html, "html.parser")
    all_links = soup.find_all("a")

    for link in all_links:
        href = link.get("href")
        if href:
            match = re.search(page_id_pattern, href)
            if match:
                found_ids.add(match.group("page_id"))

    # --- Step 2: Scan the entire raw content for any remaining page IDs ---
    # This step handles links that might be present as plain text, not in an <a> tag.
    # The existing regex is perfect for this.
    matches = re.finditer(page_id_pattern, raw_html)
    for match in matches:
        found_ids.add(match.group("page_id"))

    return list(found_ids)


@tool("extract_page_id_from_url")
def extract_page_id_from_url(url: str) -> str:
    """
    Extracts the page ID from a given Confluence URL.
    Args:
        url: The Confluence URL.
    Returns:
        The page ID as a string or an error message if not found.
    """
    pattern = r"(?:\?pageId=|/display/[^/]+/)(?P<page_id>\d+)"
    match = re.search(pattern, url)
    if match:
        return match.group("page_id")
    return "Error: Could not extract page ID from URL."


# --- 3. Build the Graph (Xây dựng biểu đồ) ---
def build_pure_langgraph_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # --- Define Nodes (Định nghĩa các nút) ---
    def get_initial_id_node(state: AgentState) -> dict:
        print("-> Trích xuất ID từ URL ban đầu...")
        page_id = extract_page_id_from_url.invoke({"url": state["user_input"]})
        if "Error" in page_id:
            return {"final_output": page_id}
        return {"urls_to_process": [page_id]}

    def fetch_page_node(state: AgentState) -> dict:
        print("-> Lấy nội dung trang...")
        urls_to_fetch = state.get("urls_to_process", [])
        processed_content = ""

        current_urls = urls_to_fetch[:]
        state["urls_to_process"] = []

        for page_id in current_urls:
            if page_id not in state.get("processed_urls", []):
                print(f"  Fetching content for page ID: {page_id}")
                content = get_confluence_page_by_id.invoke({"page_id": page_id})
                processed_content += content
                state["processed_urls"].append(page_id)

        return {"raw_content": processed_content}

    def analyze_and_find_node(state: AgentState) -> dict:
        print("-> Phân tích nội dung và tìm link mới...")
        raw_content = state["raw_content"]

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Bạn là một trợ lý thông minh chuyên phân tích tài liệu kỹ thuật. "
                    "Nhiệm vụ của bạn là trích xuất thông tin về các bảng dữ liệu, luồng chức năng và các đường link. "
                    "Nếu bạn không tìm thấy thông tin nào về 'luồng chức năng', hãy ghi rõ điều đó. "
                    "Đưa ra kết quả dưới dạng JSON với 2 trường 'is_sufficient' (boolean) . "
                    "Nếu trong đoạn summary mà thiếu thông tin nào trong các "
                    "thông tin các bảng dữ liệu, luồng chức năng thì is_sufficient là false ngược lại là true"
                    "và 'summary' (string).",
                ),
                ("user", f"Đây là nội dung tài liệu: \n\n{raw_content}"),
            ]
        )

        chain = prompt_template | llm.with_structured_output(schema=AnalysisResults)
        response = chain.invoke({"raw_content": raw_content})

        new_links = find_links.invoke({"raw_html": raw_content})

        unprocessed_links = [
            link for link in new_links if link not in state.get("processed_urls", [])
        ]

        return {"analysis_results": response, "urls_to_process": unprocessed_links}

    def decide_next_node(state: AgentState) -> str:
        print("-> Đang đưa ra quyết định...")

        is_sufficient = state.get("analysis_results", {}).get("is_sufficient")
        has_new_links = len(state.get("urls_to_process", [])) > 0
        is_within_limit = len(state.get("processed_urls", [])) <= MAX_PAGES

        if is_sufficient:
            print("-> Thông tin đủ. Kết thúc.")
            return {"next_node": "end"}
        elif not has_new_links:
            print("-> Không tìm thấy link mới. Kết thúc.")
            return {"next_node": "end"}
        elif not is_within_limit:
            print(f"-> Đã đạt giới hạn số trang ({MAX_PAGES}). Kết thúc.")
            return {"next_node": "end"}
        else:
            print("-> Thông tin chưa đủ. Tiếp tục tìm kiếm.")
            return {"next_node": "continue"}

    def final_summary_node(state: AgentState) -> dict:
        print("-> Đang tạo bản tóm tắt cuối cùng...")
        final_summary = state.get("analysis_results", {}).get(
            "summary", "Không tìm thấy thông tin phù hợp."
        )
        return {"final_output": final_summary}

    # --- Build the graph (Xây dựng biểu đồ) ---
    workflow = StateGraph(AgentState)
    workflow.add_node("get_initial_id", get_initial_id_node)
    workflow.add_node("fetch_page", fetch_page_node)
    workflow.add_node("analyze_content", analyze_and_find_node)
    workflow.add_node("decide_next", decide_next_node)
    workflow.add_node("final_summary", final_summary_node)

    workflow.set_entry_point("get_initial_id")
    workflow.add_edge("get_initial_id", "fetch_page")
    workflow.add_edge("fetch_page", "analyze_content")
    workflow.add_edge("analyze_content", "decide_next")

    workflow.add_conditional_edges(
        "decide_next",
        lambda state: state["next_node"],
        {"continue": "fetch_page", "end": "final_summary"},
    )

    workflow.add_edge("final_summary", END)

    return workflow.compile()


# --- 3. FastAPI Endpoint ---
app = FastAPI()
# Thêm CORSMiddleware vào đây, trước bất kỳ endpoint nào khác
origins = [
    "http://localhost:5173",  # Địa chỉ của ứng dụng React của bạn
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, OPTIONS, v.v.)
    allow_headers=["*"],  # Cho phép tất cả các header trong yêu cầu
)
agent_app = build_pure_langgraph_agent()


@app.post("/stream_analysis")
async def stream_analysis(request: Request):
    data = await request.json()
    url = data.get("url")

    if not url:
        return {"error": "URL is required"}

    initial_state = {
        "user_input": url,
        "raw_content": "",
        "final_output": "",
        "urls_to_process": [url],
        "processed_urls": [],
        "analysis_results": {},
    }

    # Hàm generator để phát trực tuyến các sự kiện
    async def event_generator():
        # Dùng vòng lặp 'for' thông thường để xử lý generator đồng bộ
        async for chunk in agent_app.astream(initial_state, stream_mode="updates"):
            # Mỗi chunk là một dictionary chứa thông tin về bước hiện tại
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.1)  # Thêm một khoảng dừng nhỏ

    # for event in agent_app.stream(initial_state,  stream_mode="updates"):
    #     # Mỗi 'event' ở đây là một dictionary, và chúng ta sẽ xử lý nó
    #     for key, value in event.items():
    #         if key == "__end__":
    #             yield json.dumps({"event": "end", "data": value["final_output"]}) + "\n"
    #         else:
    #             yield json.dumps({"event": key, "data": value}) + "\n"

    # StreamingResponse sẽ tự động gửi dữ liệu bất đồng bộ
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Để chạy máy chủ: uvicorn your_file_name:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)  # Thay đổi cổng nếu cần thiết
# --- 4. Main execution (Phần thực thi chính) ---
# if __name__ == "__main__":
#     app = build_pure_langgraph_agent()

#     # Người dùng nhập một đường link
#     user_initial_url = "http://localhost:8090/pages/viewpage.action?pageId=98379"

#     initial_state = {
#         "user_input": user_initial_url,
#         "raw_content": "",
#         "final_output": "",
#         "urls_to_process": [],
#         "processed_urls": [],
#         "analysis_results": {}
#     }

#     print("Starting LangGraph Agent with pure logic...")
#     final_state = app.invoke(initial_state)

#     print("\n--- Final Analysis (Phân tích cuối cùng) ---")
#     print(final_state["final_output"])
