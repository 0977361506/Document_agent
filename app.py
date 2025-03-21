from flask import Flask, request, jsonify
from functools import wraps
import asyncio
from typing import Dict, Any, Callable
import os

from server import (
    list_tools,
    confluence_search,
    confluence_get_spaces,
    confluence_get_page,
    confluence_create_page,
    confluence_update_page,
    confluence_delete_page
)

app = Flask(__name__)

def async_endpoint(f: Callable) -> Callable:
    """Decorator to handle async route handlers."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

def validate_auth() -> Dict[str, str]:
    """Validate required authentication headers."""
    confluence_url = request.headers.get('X-Confluence-URL')
    access_token = request.headers.get('X-Confluence-Token')
    
    if not confluence_url or not access_token:
        raise ValueError("Missing required headers: X-Confluence-URL and X-Confluence-Token")
        
    return {
        "confluence_url": confluence_url,
        "access_token": access_token
    }

@app.route('/api/tools', methods=['GET'])
@async_endpoint
async def get_tools():
    """List available Confluence MCP tools."""
    try:
        result = await list_tools()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/spaces', methods=['GET'])
@async_endpoint
async def get_spaces():
    """Get list of Confluence spaces."""
    try:
        auth = validate_auth()
        result = await confluence_get_spaces(**auth)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['GET'])
@async_endpoint
async def search():
    """Search Confluence content."""
    try:
        auth = validate_auth()
        query = request.args.get('query')
        if not query:
            return jsonify({"error": "Missing required parameter: query"}), 400
            
        result = await confluence_search(**auth, query=query)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pages/<space_key>/<page_id>', methods=['GET', 'PUT', 'DELETE'])
@async_endpoint
async def handle_page(space_key: str, page_id: str):
    """Handle page operations (get, update, delete)."""
    try:
        auth = validate_auth()
        
        if request.method == 'GET':
            result = await confluence_get_page(**auth, space_key=space_key, page_id=page_id)
        elif request.method == 'PUT':
            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 400
                
            content = request.json.get('content')
            if not content:
                return jsonify({"error": "Missing required field: content"}), 400
                
            result = await confluence_update_page(**auth, space_key=space_key, page_id=page_id, content=content)
        else:  # DELETE
            result = await confluence_delete_page(**auth, space_key=space_key, page_id=page_id)
            
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pages/<space_key>', methods=['POST'])
@async_endpoint
async def create_page(space_key: str):
    """Create a new page in a space."""
    try:
        auth = validate_auth()
        
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        data = request.json
        title = data.get('title')
        content = data.get('content')
        
        if not title or not content:
            return jsonify({"error": "Missing required fields: title and content"}), 400
            
        result = await confluence_create_page(**auth, space_key=space_key, title=title, content=content)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 