#!/usr/bin/env python3
"""
Flask server runner for Confluence MCP Server.
This script runs the Flask web server that provides HTTP REST API endpoints.
"""

import os
from dotenv import load_dotenv
from app import app

# Load environment variables
load_dotenv()

if __name__ == '__main__':
    # Get port from environment or default to 8090
    port = int(os.environ.get('PORT', 8090))
    
    print(f"Starting Confluence MCP Flask Server on port {port}")
    print(f"API endpoints available at:")
    print(f"  - GET  http://127.0.0.1:{port}/api/tools")
    print(f"  - GET  http://127.0.0.1:{port}/api/spaces")
    print(f"  - GET  http://127.0.0.1:{port}/api/search?query=<search_term>")
    print(f"  - GET  http://127.0.0.1:{port}/api/pages/<space_key>/<page_id>")
    print(f"  - POST http://127.0.0.1:{port}/api/pages/<space_key>")
    print(f"  - PUT  http://127.0.0.1:{port}/api/pages/<space_key>/<page_id>")
    print(f"  - DEL  http://127.0.0.1:{port}/api/pages/<space_key>/<page_id>")
    print()
    print("Required headers for API calls (except /api/tools):")
    print("  - X-Confluence-URL: Your Confluence Server URL")
    print("  - X-Confluence-Token: Your access token")
    print()
    print("Test the server:")
    print(f"  curl http://127.0.0.1:{port}/api/tools")
    print()
    
    try:
        app.run(
            host='127.0.0.1', 
            port=port, 
            debug=False
        )
    except OSError as e:
        if "address already in use" in str(e).lower() or "access permissions" in str(e).lower():
            print(f"❌ Port {port} is already in use or access denied.")
            print(f"Try a different port by setting PORT environment variable:")
            print(f"  set PORT=8091 && python run_flask_server.py")
        else:
            raise
