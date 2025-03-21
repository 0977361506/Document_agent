import requests
import json
import time

# Configuration
BASE_URL_LOCAL = "http://localhost:8080/api"
BASE_URL = "https://confluence-server-316362768082.europe-west3.run.app/api"
HEADERS = {
    "X-Confluence-URL": "https://arelis.atlassian.net/wiki",
    "X-Confluence-Token": "ATATT3xFfGF0ZnJDNUiibhs4IRWBWoBVAQXHeS-f5XAo493HD2o10DGLVosTNw5jbuAOHzdyU6mq6CIoFcOBe-CRynKMWtLmoZtlO6kGm9TAby2AZHD8g2HPDnD6KFuutA3369Dmzzc4936Ha82aUh0XLFhgr58bg5EbImndKcH4FaYfGVsO2NE=050B456E",
    "Content-Type": "application/json"
}

def print_response(response, operation):
    """Pretty print the response."""
    print(f"\n=== {operation} ===")
    print(f"Status Code: {response.status_code}")
    try:
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    except:
        print("Raw Response:", response.text)
    print("=" * (len(operation) + 8))

def test_deployed_api():
    """Test all API endpoints on the deployed Cloud Run instance."""
    print(f"\n🚀 Testing deployed API at: {BASE_URL}")
    
    # 1. List Tools
    print("\nTesting List Tools...")
    response = requests.get(f"{BASE_URL}/tools", headers=HEADERS)
    print_response(response, "List Tools")

    # 2. List Spaces
    print("\nTesting Get Spaces...")
    response = requests.get(f"{BASE_URL}/spaces", headers=HEADERS)
    print_response(response, "Get Spaces")

    # 3. Search Content
    print("\nTesting Search...")
    query = "Arelis Web Platform Architecture"
    response = requests.get(f"{BASE_URL}/search", headers=HEADERS, params={"query": query})
    print_response(response, "Search")

    # 4. Create, Read, Update, Delete Page
    print("\nTesting Page Operations...")
    
    # Create Page
    space_key = "SD"
    test_page_title = f"Test Page {int(time.time())}"
    create_data = {
        "title": test_page_title,
        "content": "<p>This is a test page created via Cloud Run API</p>"
    }
    
    response = requests.post(
        f"{BASE_URL}/pages/{space_key}",
        headers=HEADERS,
        json=create_data
    )
    print_response(response, "Create Page")
    
    # Extract page ID from response
    page_data = response.json()
    page_id = None
    for content in page_data.get('content', []):
        if content.get('type') == 'json':
            page_id = content.get('data', {}).get('content_id')
            break
    
    if page_id:
        # Read Page
        response = requests.get(
            f"{BASE_URL}/pages/{space_key}/{page_id}",
            headers=HEADERS
        )
        print_response(response, "Read Page")
        
        # Update Page
        update_data = {
            "content": "<p>This is an updated test page via Cloud Run API</p>"
        }
        response = requests.put(
            f"{BASE_URL}/pages/{space_key}/{page_id}",
            headers=HEADERS,
            json=update_data
        )
        print_response(response, "Update Page")
        
        # Delete Page
        response = requests.delete(
            f"{BASE_URL}/pages/{space_key}/{page_id}",
            headers=HEADERS
        )
        print_response(response, "Delete Page")
    else:
        print("Failed to get page ID from create response")

def test_local_api():
    """Test all API endpoints on the local instance."""
    print(f"\n🖥️ Testing local API at: {BASE_URL_LOCAL}")
    
    # Same structure as test_deployed_api but using BASE_URL_LOCAL
    # This is useful for comparing local vs deployed behavior
    # ... (Implementation similar to test_deployed_api)

if __name__ == "__main__":
    # Comment/uncomment as needed
    test_deployed_api()
    # test_local_api()  # Uncomment to test local server 