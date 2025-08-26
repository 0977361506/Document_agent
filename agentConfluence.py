import asyncio
import datetime
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
from openai import BaseModel
from pydantic import conlist

# Load environment variables
load_dotenv()


# --- 1. Define State (Trạng thái) ---
class AgentState(TypedDict):
    user_input: str  # Giờ đây có thể là URL Jira hoặc URL template
    jira_url: str
    template_url: str
    topic_of_interest: str
    raw_content: str
    final_output: str  # Đây là bản tóm tắt cuối cùng từ luồng cũ
    urls_to_process: List[str]
    processed_urls: List[str]
    analysis_results: Dict[str, Any]
    error: str
    template_content: str
    template_params: List[str]  # Ví dụ: ['flow', 'database_tables', 'api_spec']
    filled_params: Dict[str, str]


class AnalysisResults(TypedDict):
    is_sufficient: bool
    summary: str


class InfoConfluence(TypedDict):
    topic: str
    raw_content: str


class TemplateParams(TypedDict):
    params: List[str]


class TemplateContent(TypedDict):
    templateContent: str


# Định nghĩa cấu trúc cho một cột trong bảng
class Cot(TypedDict):
    ten: str
    kieu: str
    mo_ta: str


# Định nghĩa cấu trúc cho một bảng dữ liệu
class Bang(TypedDict):
    ten_bang: str
    mo_ta: str
    cot: List[Cot]


# Định nghĩa cấu trúc cho một API endpoint
class API(TypedDict):
    ten_api: str
    endpoint: str
    mo_ta: str
    request_body: str
    response_success_mo_ta: str
    response_success_body: str
    response_error_mo_ta: str
    response_error_body: str


# Định nghĩa cấu trúc tổng thể cho toàn bộ tài liệu
class ParamValues(TypedDict):
    ten_chuc_nang: str
    mo_ta_chuc_nang: str
    so_do_hoat_dong: str
    so_do_tuan_tu: str
    so_do_quan_he_entity: str
    danh_sach_bang: List[Bang]
    danh_sach_api: List[API]


# --- 2. Define Tools (Công cụ) ---
@tool
def get_template_content(template_url: str) -> str:
    """
    Fetches the raw content of a Confluence template page.
    Args:
        template_url: The URL of the Confluence template.
    Returns:
        The raw HTML content of the template as a string.
    """
    # Use existing extract_page_id_from_url and get_confluence_page_by_id
    page_id = extract_page_id_from_url.invoke({"url": template_url})
    confluence_info = get_confluence_page_by_id.invoke({"page_id": page_id})
    return confluence_info["raw_content"]


# tools.py
@tool
def create_new_confluence_page(
    space_key: str, title: str, content: str, parent_page_id: str = None
) -> str:
    """
    Creates a new page in a specified Confluence space.
    Args:
        space_key: The key of the Confluence space (e.g., 'DEV').
        title: The title of the new page.
        content: The content of the new page in Confluence Storage Format.
        parent_page_id: Optional parent page ID (if you want to create under a specific page).
    Returns:
        The URL of the newly created page.
    """
    confluence_url = "http://localhost:8090"  # ví dụ: http://localhost:8090
    # username = os.getenv("CONFLUENCE_USERNAME")  # user login Confluence Server
    # password = os.getenv("CONFLUENCE_PASSWORD")  # password hoặc PAT của user

    # if not confluence_url or not username or not password:
    #     raise ValueError("Confluence URL, username or password is not set.")

    api_endpoint = f"{confluence_url}/rest/api/content/"

    payload = {
        "type": "page",
        "title": title,
        "space": {"key": space_key},
        "body": {"storage": {"value": content, "representation": "storage"}},
    }

    # Nếu muốn tạo page con trong 1 page cha
    if parent_page_id:
        payload["ancestors"] = [{"id": parent_page_id}]

    try:
        api_token = os.getenv("CONFLUENCE_API_TOKEN")
        headers = {"Accept": "application/json", "Authorization": f"Bearer {api_token}"}
        response = requests.post(
            api_endpoint,
            headers=headers,
            json=payload,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Error creating Confluence page: {response.status_code} - {response.text}"
            )

        new_page_data = response.json()
        new_page_id = new_page_data["id"]
        return f"{confluence_url}/pages/viewpage.action?pageId={new_page_id}"
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error creating Confluence page: {e}")


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
                        "và 'summary' (string). is_sufficient sẽ là true nếu bạn tìm thấy đủ thông tin về luồng chức năng hoặc các bảng dữ liệu liên quan đến '{topic}', "
                        "ngược lại nếu thiếu một thông tin nào đó là false."
                        "Nếu không tìm thấy bất kỳ thông tin nào liên quan đến '{topic}' trong nội dung, hãy đặt summary là 'Không tìm thấy thông tin liên quan đến {topic}.'",
                    ),
                    ("user", "Đây là nội dung tài liệu:\n\n{raw_content}"),
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

    # Sửa đổi hàm final_summary_node
    def final_summary_node(state: AgentState) -> dict:
        print("-> Đang tạo bản tóm tắt cuối cùng...")
        final_summary = state.get("analysis_results", {}).get(
            "summary", "Không tìm thấy thông tin phù hợp."
        )
        topic = state.get("topic_of_interest", "một chủ đề không xác định")

        # Nếu có lỗi, trả về thông báo lỗi
        if state.get("error"):
            return {"final_output": state["final_output"]}

        # Nếu phân tích ban đầu không đủ, yêu cầu LLM suy luận thêm
        is_sufficient = state.get("analysis_results", {}).get("is_sufficient", False)
        if not is_sufficient:
            print("-> Thông tin chưa đủ. Yêu cầu AI suy luận và bổ sung...")

            # Tạo prompt mới để yêu cầu LLM suy luận
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", temperature=0.7
            )  # Tăng temperature để có khả năng sáng tạo hơn
            reasoning_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Dựa trên các thông tin đã tìm thấy sau đây về chủ đề '{topic}', "
                        "hãy suy luận và bổ sung thêm các chi tiết còn thiếu một cách hợp lý. "
                        "Nếu không có thông tin cụ thể về 'luồng chức năng' hoặc 'cấu trúc bảng', "
                        "hãy suy luận và bổ sung thêm các chi tiết còn thiếu một cách hợp lý."
                        "Không tạo ra thông tin sai lệch, chỉ suy luận dựa trên ngữ cảnh chung và kiến thức của bạn. "
                        "Trả lại toàn bộ thông tin đã tìm thấy và phần suy luận của bạn thành một bản báo cáo hoàn chỉnh. Không giải thích gì thêm",
                    ),
                    ("user", "Thông tin đã tìm thấy:\n\n{final_summary}"),
                ]
            )

            reasoning_chain = reasoning_prompt | llm
            try:
                full_summary_response = reasoning_chain.invoke(
                    {"final_summary": final_summary, "topic": topic}
                )
                final_summary = (
                    full_summary_response.content
                )  # Lấy nội dung từ phản hồi của LLM
            except Exception as e:
                # Xử lý nếu LLM gọi thất bại
                print(f"Lỗi khi gọi LLM để suy luận: {e}")
                final_summary = (
                    final_summary
                    + "\n\n(Không thể suy luận và bổ sung thêm thông tin do lỗi.)"
                )

        return {"final_output": final_summary}

    def get_template_and_params_node(state: AgentState) -> dict:
        print("-> Lấy nội dung template và phân tích các tham số...")
        try:
            page_id = extract_page_id_from_url.invoke({"url": state["template_url"]})
            confluence_info = get_confluence_page_by_id.invoke({"page_id": page_id})
            template_content = confluence_info["raw_content"]
            # Dùng LLM để phân tích template và tìm các tham số cần điền
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Bạn là một trợ lý thông minh chuyên phân tích tài liệu kỹ thuật. "
                        "Hãy đọc nội dung template sau và liệt kê tất cả các tham số trong dấu ngoặc [] hoặc phần cần điền vào. Ví dụ [mo_ta_chuc_nang]"
                        "Trả về một danh sách các chuỗi param.",
                    ),
                    ("user", "Đây là nội dung template:\n\n{template_content}"),
                ]
            )
            # Sử dụng một custom parser để xử lý kết quả
            chain = prompt | llm.with_structured_output(schema=TemplateParams)
            response = chain.invoke({"template_content": template_content})

            # Chuyển đổi response thành list of strings
            # Có thể cần một bước xử lý phức tạp hơn tùy thuộc vào LLM output format
            params = response["params"]

            return {"template_content": template_content, "template_params": params}
        except Exception as e:
            return {"error": f"Error in get_template_and_params_node: {e}"}

    def extract_and_fill_params_node(state: AgentState) -> dict:
        print("-> Trích xuất thông tin từ bản tóm tắt và điền vào tham số...")
        try:
            final_summary = state["final_output"]
            template_params = state["template_params"]
            topic = state["topic_of_interest"]

            # Dùng LLM để phân tích bản tóm tắt
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Bạn là một trợ lý phân tích hệ thống. Dựa vào nội dung tài liệu yêu cầu chức năng (Business Requirement Document) được cung cấp, "
                        "hãy phân tích và trích xuất tất cả các thông tin cần thiết. Sau đó, trả về một đối tượng JSON có cấu trúc tuân thủ nghiêm ngặt theo schema đã định sẵn. "
                        "Nếu một trường thông tin không có trong tài liệu, hãy điền giá trị là 'null'."
                        "\n**Lưu ý:**\n"
                        "1.  **ten_chuc_nang**: Tên ngắn gọn của chức năng.\n"
                        "2.  **mo_ta_chuc_nang**: Tổng hợp mô tả chức năng một cách súc tích.\n"
                        "3.  **so_do_hoat_dong**: Mô tả luồng hoạt động chính, sử dụng ký hiệu `->` hoặc liệt kê các bước tuần tự.\n"
                        "4.  **so_do_tuan_tu**: Diễn tả luồng tương tác giữa các thành phần (người dùng, giao diện, API, microservice) một cách tuần tự.\n"
                        "5.  **so_do_quan_he_entity**: Liệt kê các entity và mối quan hệ của chúng (ví dụ: 'USER có nhiều ORDER').\n"
                        "6.  **danh_sach_bang**: Liệt kê tất cả các bảng (entities) liên quan, với tên bảng, mô tả, và danh sách các cột cùng kiểu dữ liệu tương ứng.\n"
                        "7.  **danh_sach_api**: Liệt kê tất cả các API được định nghĩa, bao gồm tên, endpoint, mô tả, và cấu trúc request/response body (định dạng chuỗi JSON).\n"
                        "8.  Nếu một trường thông tin không có trong tài liệu, hãy điền giá trị là 'null'.",
                    ),
                    (
                        "user",
                        "topic:{topic}"
                        "Các tham số cần điền: {template_params}\n\n"
                        "Bản tóm tắt nội dung:\n{final_summary}",
                    ),
                ]
            )

            chain = prompt | llm.with_structured_output(schema=ParamValues)
            response = chain.invoke(
                {
                    "final_summary": final_summary,
                    "template_params": template_params,
                    "topic": topic,
                }
            )
            filled_params = response

            return {"filled_params": filled_params}
        except Exception as e:
            return {"error": f"Error in extract_and_fill_params_node: {e}"}

    def create_document_node(state: AgentState) -> dict:
        print("-> Tạo tài liệu hoàn chỉnh và đẩy lên Confluence...")
        try:
            template_content = state["template_content"]
            filled_params = state["filled_params"]
            topic = state["topic_of_interest"]

            # Dùng LLM để điền các tham số vào template
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            # prompt = ChatPromptTemplate.from_messages(
            #     [
            #         (
            #             "system",
            #             "Bạn là một trợ lý thông minh. Hãy sử dụng các giá trị đã trích xuất "
            #             "để điền vào template sau đây, tạo ra một tài liệu hoàn chỉnh.",
            #         ),
            #         (
            #             "user",
            #             "Template:\n{template_content}\n\n"
            #             "Các giá trị:\n{filled_params}",
            #         ),
            #     ]
            # )
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """Bạn là một trợ lý ảo chuyên về phân tích và thiết kế hệ thống. Dựa vào nội dung template tài liệu phân tích thiết kế chức năng và dữ liệu dưới đây, hãy tạo ra một tài liệu hoàn chỉnh.
                        Kết quả trả về kiểu string.

                        **Quy tắc:**
                        1. Đối với các sơ đồ (Hoạt động, Tuần tự, ERD), hãy tạo mã Mermaid phù hợp với dữ liệu được cung cấp.
                        2. **Rất quan trọng:** Bọc mã Mermaid đã tạo vào trong macro Confluence sau đây. Đảm bảo mã Mermaid được đặt bên trong thẻ CDATA.
                            <ac:structured-macro ac:name="mermaiddiagram" ac:schema-version="1" ac:macro-id="9b4ed081-49b0-4cfb-8a65-331b6b4906d2"><ac:parameter ac:name="" /><ac:plain-text-body><![CDATA[ [MÃ MERMAID CỦA BẠN] ]]></ac:plain-text-body></ac:structured-macro>
                        3. Đối với phần 'Danh sách Bảng Dữ liệu' và 'Mô tả API', hãy tạo các bảng và danh sách chi tiết dựa trên dữ liệu.
                        """,
                    ),
                    (
                        "user",
                        "Template:\n{template_content}\n\n"
                        "Các giá trị đã điền:\n{filled_params}",
                    ),
                ]
            )
            chain = prompt | llm
            final_document_content = chain.invoke(
                {"template_content": template_content, "filled_params": filled_params}
            ).content
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Lấy space key từ environment variable hoặc input
            marco = """<ac:structured-macro ac:name="mermaiddiagram" ac:schema-version="1" ac:macro-id="9b4ed081-49b0-4cfb-8a65-331b6b4906d2"><ac:parameter ac:name="" /><ac:plain-text-body><![CDATA[flowchart TD
                    A[Christmas] -->|Get money| B(Go shopping)
                    B --> C{Let me think}
                    C -->|One| D[Laptop]
                    C -->|Two| E[iPhone]
                    C -->|Three| F[fa:fa-car Car]]]></ac:plain-text-body></ac:structured-macro>"""
            space_key = "DG"
            new_page_url = create_new_confluence_page.invoke(
                {
                    "space_key": space_key,
                    "title": f"PTTK - {topic} ({current_time})",
                    "content": final_document_content,
                }
            )

            return {
                "final_output": f"Tài liệu PTTK đã được tạo thành công: {new_page_url}"
            }
        except Exception as e:
            return {"error": f"Error in create_document_node: {e}"}

    # --- Build the graph (Xây dựng biểu đồ) ---
    workflow = StateGraph(AgentState)
    workflow.add_node("get_template_and_params", get_template_and_params_node)
    workflow.add_node(
        "get_initial_id", get_initial_id_node
    )  # Node này sẽ chỉ lấy ID Jira
    workflow.add_node("fetch_page", fetch_page_node)
    workflow.add_node("analyze_content", analyze_and_find_node)
    workflow.add_node("final_summary", final_summary_node)
    workflow.add_node("extract_and_fill", extract_and_fill_params_node)
    workflow.add_node("create_document", create_document_node)

    # Thiết lập entry point và các cạnh
    workflow.set_entry_point("get_template_and_params")
    workflow.add_edge("get_template_and_params", "get_initial_id")
    workflow.add_edge("get_initial_id", "fetch_page")
    workflow.add_edge("fetch_page", "analyze_content")
    workflow.add_conditional_edges(
        "analyze_content",
        decide_next_node,  # Logic này giữ nguyên
        {
            "continue": "fetch_page",
            "end": "final_summary",
        },
    )
    workflow.add_edge("final_summary", "extract_and_fill")
    workflow.add_edge("extract_and_fill", "create_document")
    workflow.add_edge("create_document", END)

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


@app.post("/api/v1/stream_analysis")
async def stream_analysis(request: Request):
    data = await request.json()
    url = data.get("url")
    if not url:
        return {"error": "URL is required"}
    # initial_state = {
    #     "user_input": url,
    #     "raw_content": "",
    #     "final_output": "",
    #     "urls_to_process": [],
    #     "processed_urls": [],
    #     "analysis_results": {},
    #     "error": None,
    # }

    # Initialize the state with all the new fields
    initial_state = {
        "user_input": url,  # user_input is now the jira_url
        "template_url": "http://localhost:8090/pages/viewpage.action?pageId=1015810",
        "topic_of_interest": "",
        "raw_content": "",
        "final_output": "",
        "urls_to_process": [],
        "processed_urls": [],
        "analysis_results": {},
        "error": None,
        "template_content": "",
        "template_params": [],
        "filled_params": {},
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
