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


# --- 1. Define State (Trạng thái) ---
class AgentState(TypedDict):
    user_input: str
    topic_of_interest: str
    raw_content: str
    final_output: str
    urls_to_process: List[str]
    processed_urls: List[str]
    analysis_results: Dict[str, Any]
    error: str


class AnalysisResults(TypedDict):
    is_sufficient: bool
    summary: str


class InfoConfluence(TypedDict):
    topic: str
    raw_content: str


# --- 2. Define Tools (Công cụ) ---
@tool
def get_confluence_page_by_id(page_id: str) -> InfoConfluence:
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
        raise ValueError(
            "Confluence URL or API token is not set in environment variables."
        )
    api_endpoint = f"{confluence_url}/rest/api/content/{page_id}?expand=body.storage"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_token}"}
    try:
        response = requests.get(api_endpoint, headers=headers)
        response.raise_for_status()
        data = response.json()
        raw_content = data["body"]["storage"]["value"]
        topic = data["title"]
        return {"topic": topic, "raw_content": raw_content}
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching page from Confluence: {e}")


@tool("find_links")
def find_links(raw_html: str) -> List[str]:
    """
    Searches for Confluence page links (URLs or page IDs) within raw HTML content,
    prioritizing <a> tags and then scanning the rest of the text.
    Returns:
        A list of unique Confluence page IDs found in the content.
    """
    found_ids = set()
    page_id_pattern = r"(?:\?pageId=|/display/[^/]+/)(?P<page_id>\d+)"
    soup = BeautifulSoup(raw_html, "html.parser")
    all_links = soup.find_all("a")
    for link in all_links:
        href = link.get("href")
        if href:
            match = re.search(page_id_pattern, href)
            if match:
                found_ids.add(match.group("page_id"))
    matches = re.finditer(page_id_pattern, raw_html)
    for match in matches:
        found_ids.add(match.group("page_id"))
    return list(found_ids)


@tool("extract_page_id_from_url")
def extract_page_id_from_url(url: str) -> str:
    """
    Extracts the page ID from a given Confluence URL.
    """
    pattern = r"(?:\?pageId=|/display/[^/]+/)(?P<page_id>\d+)"
    match = re.search(pattern, url)
    if match:
        return match.group("page_id")
    raise ValueError("Could not extract page ID from URL.")


# --- 3. Build the Graph (Xây dựng biểu đồ) ---
def build_pure_langgraph_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # --- Define Nodes (Định nghĩa các nút) ---

    def get_initial_id_node(state: AgentState) -> dict:
        try:
            print("-> Trích xuất ID và tiêu đề từ URL ban đầu...")

            page_id = extract_page_id_from_url.invoke({"url": state["user_input"]})
            confluence_info = get_confluence_page_by_id.invoke({"page_id": page_id})

            # Lấy giá trị chuỗi từ dictionary
            topic = confluence_info.get("topic")

            return {
                "urls_to_process": [page_id],
                # "raw_content": initial_content,
                "topic_of_interest": topic,
            }
        except Exception as e:
            return {"final_output": f"Error: {e}", "error": f"{e}"}

    def fetch_page_node(state: AgentState) -> dict:
        print("-> Lấy nội dung trang...")
        urls_to_fetch = state.get("urls_to_process", [])

        cumulative_content = state.get("raw_content", "")

        current_urls = urls_to_fetch[:]
        state["urls_to_process"] = []

        for page_id in current_urls:
            if page_id not in state.get("processed_urls", []):
                try:
                    print(f"   Fetching content for page ID: {page_id}")
                    content = get_confluence_page_by_id.invoke({"page_id": page_id})
                    cumulative_content += (
                        f"\n\n--- Content from page {page_id} ---\n\n"
                        + content.get("raw_content")
                    )
                    state["processed_urls"].append(page_id)
                except Exception as e:
                    return {"final_output": f"Error: {e}", "error": f"{e}"}

        return {"raw_content": cumulative_content}

    def analyze_and_find_node(state: AgentState) -> dict:
        try:
            print("-> Phân tích nội dung và tìm link mới...")
            raw_content = state["raw_content"]
            topic = state.get("topic_of_interest", "một chủ đề không xác định")

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Bạn là một trợ lý thông minh chuyên phân tích tài liệu kỹ thuật. "
                        "Nhiệm vụ của bạn là tổng hợp thông tin từ các tài liệu liên quan đến chủ đề '{topic}'. "
                        "Hãy tìm thông tin chi tiết về các bảng dữ liệu (bao gồm tên bảng và các thông tin về cột trong bảng), luồng chức năng và các đường link liên quan. "
                        "Nếu không tìm thấy thông tin nào về 'luồng chức năng', hãy ghi rõ. "
                        "Đưa ra kết quả dưới dạng JSON với 2 trường 'is_sufficient' (boolean) "
                        "và 'summary' (string). is_sufficient sẽ là true nếu bạn tìm thấy đủ thông tin về luồng chức năng hoặc các bảng dữ liệu liên quan đến '{topic}', ngược lại là false."
                        "Nếu không tìm thấy bất kỳ thông tin nào liên quan đến '{topic}' trong nội dung, hãy đặt summary là 'Không tìm thấy thông tin liên quan đến {topic}.'",
                    ),
                    ("user", f"Đây là nội dung tài liệu:\n\n{raw_content}"),
                ]
            )
            chain = prompt_template | llm.with_structured_output(schema=AnalysisResults)
            response = chain.invoke({"raw_content": raw_content, "topic": topic})

            new_links = find_links.invoke({"raw_html": raw_content})

            unprocessed_links = [
                link
                for link in new_links
                if link not in state.get("processed_urls", [])
            ]

            # Reset raw_content để tránh tổng hợp lại từ đầu
            state["raw_content"] = ""

            return {"analysis_results": response, "urls_to_process": unprocessed_links}
        except Exception as e:
            return {"final_output": f"Error: {e}", "error": f"{e}"}

    # Cấu hình hàm điều kiện đúng cách
    def decide_next_node(state: AgentState) -> str:
        print("-> Đang đưa ra quyết định...")

        if state.get("error"):
            print("-> Phát hiện lỗi, dừng lại.")
            return "end"

        is_sufficient = state.get("analysis_results", {}).get("is_sufficient")
        has_new_links = len(state.get("urls_to_process", [])) > 0

        if is_sufficient:
            print("-> Thông tin đủ. Kết thúc.")
            return "end"
        elif not has_new_links:
            print("-> Không tìm thấy link mới. Kết thúc.")
            return "end"
        else:
            print("-> Thông tin chưa đủ. Tiếp tục tìm kiếm.")
            return "continue"

    def final_summary_node(state: AgentState) -> dict:
        print("-> Đang tạo bản tóm tắt cuối cùng...")
        final_summary = state.get("analysis_results", {}).get(
            "summary", "Không tìm thấy thông tin phù hợp."
        )
        # Nếu có lỗi, trả về thông báo lỗi thay vì tóm tắt
        if state.get("error"):
            return {"final_output": state["final_output"]}
        return {"final_output": final_summary}

    # --- Build the graph (Xây dựng biểu đồ) ---
    workflow = StateGraph(AgentState)
    workflow.add_node("get_initial_id", get_initial_id_node)
    workflow.add_node("fetch_page", fetch_page_node)
    workflow.add_node("analyze_content", analyze_and_find_node)
    # Không cần add_node("decide_next", ...) nữa
    workflow.add_node("final_summary", final_summary_node)

    workflow.set_entry_point("get_initial_id")
    workflow.add_edge("get_initial_id", "fetch_page")
    workflow.add_edge("fetch_page", "analyze_content")

    # Sử dụng analyze_content làm node trước node điều kiện
    workflow.add_conditional_edges(
        "analyze_content",
        decide_next_node,  # <--- GỌI HÀM CONDITION TRỰC TIẾP
        {
            "continue": "fetch_page",
            "end": "final_summary",
        },
    )

    workflow.add_edge("final_summary", END)

    return workflow.compile()


# --- 4. FastAPI Endpoint ---
app = FastAPI()
# Thêm CORSMiddleware
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        "urls_to_process": [],
        "processed_urls": [],
        "analysis_results": {},
        "error": None,
    }

    async def event_generator():
        try:
            async for chunk in agent_app.astream(initial_state, stream_mode="updates"):
                if "error" in chunk and chunk["error"]:
                    yield f"data: {json.dumps(chunk)}\n\n"
                    break
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.1)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'final_output': f'An unexpected error occurred: {e}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
