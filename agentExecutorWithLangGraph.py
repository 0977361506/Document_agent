import os
import re
from typing import Any, Dict, List, TypedDict

import requests
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv()

# Giới hạn số lần lặp tối đa để tránh vòng lặp vô hạn
MAX_AGENT_ITERATIONS = 5


# --- 1. Define State (Trạng thái) ---
class AgentState(TypedDict):
    user_input: str
    raw_content: str
    final_output: str
    processed_urls: List[str]
    agent_output: Any
    iterations: int  # Thêm trường để đếm số lần agent lặp
    next_node: str  # Thêm trường để lưu trạng thái của nút tiếp theo


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
    Searches for Confluence page links (URLs or page IDs) within raw HTML content.
    Returns:
        A list of unique Confluence page IDs found in the content.
    """
    pattern = r"(?:\?pageId=|/display/[^/]+/)(?P<page_id>\d+)"
    found_ids = set()
    matches = re.finditer(pattern, raw_html)
    for match in matches:
        found_ids.add(match.group("page_id"))
    return list(found_ids)


# --- 3. Xây dựng Agent (Build the Agent) ---
def create_confluence_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    tools = [get_confluence_page_by_id, find_links]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Bạn là một trợ lý thông minh chuyên phân tích tài liệu Confluence. "
                    "Nhiệm vụ của bạn là lấy và phân tích **TẤT CẢ** thông tin liên quan đến các **bảng dữ liệu, luồng chức năng và các đường link**. "
                    "Nếu bạn không tìm thấy thông tin liên quan đến chức năng, hãy **ghi rõ điều đó** trong báo cáo cuối cùng của bạn. "
                    "Nếu bạn cần thêm thông tin để có một phân tích đầy đủ, bạn phải tự động tìm kiếm các link trong tài liệu và truy cập các trang đó. "
                    "Hãy trả lời bằng tiếng Việt và **đảm bảo rằng báo cáo của bạn thể hiện rõ những gì đã tìm thấy và những gì còn thiếu**."
                ),
            ),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


# --- 4. Xây dựng biểu đồ (Build the Graph) ---
def build_agentic_graph():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # Nút Agent chính - sử dụng AgentExecutor
    agent_executor_node = create_confluence_agent()

    # Định nghĩa các nút cho LangGraph
    def agent_node(state: AgentState) -> dict:
        print(f"--- Iteration {state['iterations'] + 1} ---")

        # Gọi Agent Executor để suy luận và hành động
        agent_output = agent_executor_node.invoke({"input": state["user_input"]})
        print(f"--- agent_output {state['agent_output'] } ---")
        # Cập nhật trạng thái và tăng số lần lặp
        return {"agent_output": agent_output, "iterations": state["iterations"] + 1}

    def decide_to_continue_node(state: AgentState) -> str:
        print(f"Vào đây")
        # Kiểm tra xem agent đã đưa ra câu trả lời cuối cùng chưa
        if (
            isinstance(state["agent_output"], dict)
            and "output" in state["agent_output"]
        ):
            print("-> Agent has provided a final answer. Ending.")
            return {"next_node": "end"}

        # Kiểm tra giới hạn lặp
        if state["iterations"] >= MAX_AGENT_ITERATIONS:
            print(f"-> Reached maximum iterations of {MAX_AGENT_ITERATIONS}. Ending.")
            return {"next_node": "end"}

        print("-> Agent needs more information. Continuing the loop.")
        return {"next_node": "continue"}

    def generate_final_summary_node(state: AgentState) -> dict:
        final_summary = state["agent_output"].get(
            "output", "Không thể tìm thấy thông tin phù hợp sau khi phân tích."
        )
        return {"final_output": final_summary}

    # Xây dựng biểu đồ
    workflow = StateGraph(AgentState)
    workflow.add_node("run_agent", agent_node)
    workflow.add_node("decide_next", decide_to_continue_node)
    workflow.add_node("generate_summary", generate_final_summary_node)

    workflow.set_entry_point("run_agent")

    workflow.add_edge("run_agent", "decide_next")

    workflow.add_conditional_edges(
        "decide_next",
        lambda state: state["next_node"],
        {"continue": "run_agent", "end": "generate_summary"},
    )

    workflow.add_edge("generate_summary", END)

    return workflow.compile()


# --- 5. Main execution (Phần thực thi chính) ---
if __name__ == "__main__":
    app = build_agentic_graph()

    user_initial_request = "Phân tích nội dung và các bảng chức năng liên quan từ trang Confluence có ID là 98379."

    initial_state = {
        "user_input": user_initial_request,
        "raw_content": "",
        "final_output": "",
        "processed_urls": [],
        "agent_output": {},
        "iterations": 0,
        "next_node": "",
    }

    print("Starting Agentic Confluence Graph...")
    # # Show workflow
    # display(Image(app.get_graph().draw_mermaid_png()))
    final_state = app.invoke(initial_state)

    print("\n--- Final Analysis (Phân tích cuối cùng) ---")
    print(final_state["final_output"])
