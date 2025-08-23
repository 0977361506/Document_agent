# Confluence MCP Server

[![MCP Reviewed](https://img.shields.io/badge/MCP%20Reviewed-✓-blue)](https://mcpreview.com/mcp-servers/pawankumar94/confluence-mcp-server)

A Model Context Protocol (MCP) server implementation for Atlassian Confluence Server. This server provides a set of tools for interacting with Confluence Server through the MCP protocol, allowing AI agents to seamlessly work with Confluence content. Built with Flask for easy deployment to Cloud Run.

## Features

- Search pages and spaces using Confluence Query Language (CQL)
- List all available Confluence spaces
- Create, read, update, and delete Confluence pages
- Rich metadata support for Confluence resources
- Flask-based server for Cloud Run deployment
- MCP tools for AI agent integration
- **Confluence Server support** with Personal Access Token authentication
- Default token configuration for quick setup

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with the following variables:

```env
CONFLUENCE_URL=https://your-confluence-server.com
CONFLUENCE_ACCESS_TOKEN=NjgwODAxODIzMzIwOvsgoNEGsolZUPSWL7PT3TMvOv6m
PORT=8080  # Optional, defaults to 8080
```

To get an access token for Confluence Server:
1. Log in to your Confluence Server instance as an administrator
2. Go to Administration > General Configuration > Personal Access Tokens
3. Create a new Personal Access Token and copy it
4. Alternatively, use the provided default token: NjgwODAxODIzMzIwOvsgoNEGsolZUPSWL7PT3TMvOv6m

## Migration from Confluence Cloud

This server has been updated to work with **Confluence Server** instead of Confluence Cloud. Key changes include:

- **Authentication**: Now uses Personal Access Tokens instead of username/password
- **API Endpoints**: Configured for Confluence Server API endpoints
- **Default Token**: Includes a pre-configured token for quick testing
- **Cloud Flag**: Set to `false` for Confluence Server compatibility

If you need to revert to Confluence Cloud, change the `cloud=False` to `cloud=True` in the `create_confluence_client` function and update the authentication method.

## Available Tools

The server provides the following MCP tools:

### 1. Search Content
```python
@tool("search_confluence")
def search(query: str) -> Dict[str, Any]
```

### 2. Get Spaces
```python
@tool("get_spaces")
def get_spaces() -> Dict[str, Any]
```

### 3. Get Page Content
```python
@tool("get_page_content")
def get_page_content(space_key: str, page_id: str) -> Dict[str, Any]
```

### 4. Create Page
```python
@tool("create_page")
def create_page(space_key: str, title: str, content: str) -> Dict[str, Any]
```

### 5. Update Page
```python
@tool("update_page")
def update_page(space_key: str, page_id: str, content: str) -> Dict[str, Any]
```

### 6. Delete Page
```python
@tool("delete_page")
def delete_page(space_key: str, page_id: str) -> Dict[str, Any]
```

## Running Locally

Run the server locally:
```bash
python example.py
```

The server will start on http://localhost:8080

## Cloud Run Deployment

1. Build the Docker image:
```bash
docker build -t confluence-mcp .
```

2. Tag and push to Google Container Registry:
```bash
docker tag confluence-mcp gcr.io/[PROJECT-ID]/confluence-mcp
docker push gcr.io/[PROJECT-ID]/confluence-mcp
```

3. Deploy to Cloud Run:
```bash
gcloud run deploy confluence-mcp \
  --image gcr.io/[PROJECT-ID]/confluence-mcp \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="CONFLUENCE_URL=[YOUR_URL],CONFLUENCE_ACCESS_TOKEN=[YOUR_TOKEN]"
```

## Error Handling

All tools include proper error handling and will return appropriate error messages in the response. The response format includes:
- Success case: Relevant data in the specified format
- Error case: `{"error": "error message"}`

## Security Considerations

1. Always use environment variables for sensitive data
2. Consider using Cloud Run's built-in secret management
3. Implement proper authentication for your endpoints
4. Keep your Confluence access token secure

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 