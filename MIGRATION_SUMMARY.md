# Migration Summary: Confluence Cloud to Confluence Server

## Overview
This document summarizes the changes made to migrate the Confluence MCP Server from Confluence Cloud to Confluence Server authentication.

## Key Changes Made

### 1. Authentication Method
**Before (Confluence Cloud):**
```python
Confluence(
    url=url,
    username="pawan.kumar@arelis.digital",
    password=token,
    cloud=True,
    timeout=30
)
```

**After (Confluence Server):**
```python
Confluence(
    url=url,
    token=token,
    cloud=False,
    timeout=30
)
```

### 2. Default Token Configuration
- Added default token constant: `NjgwODAxODIzMzIwOvsgoNEGsolZUPSWL7PT3TMvOv6m`
- Implemented fallback logic to use default token if none provided
- Added logging when default token is used

### 3. Documentation Updates
- Updated all example URLs from `https://your-domain.atlassian.net/wiki` to `https://your-confluence-server.com`
- Updated all example tokens to use the new default token
- Added migration section in README.md
- Updated configuration instructions for Confluence Server

### 4. Files Modified
- `server.py`: Main authentication and API logic
- `README.md`: Documentation and examples
- `app.py`: Minor cleanup of unused imports
- Created `test_confluence_server.py`: Test script for verification
- Created `.env.example`: Configuration template

### 5. New Features
- Automatic fallback to default token
- Enhanced logging for authentication
- Test script for validation
- Configuration template

## Testing
Use the provided `test_confluence_server.py` script to verify the integration:

1. Update the `confluence_url` variable with your actual server URL
2. Uncomment the test execution line
3. Run: `python test_confluence_server.py`

## Configuration
1. Copy `.env.example` to `.env`
2. Update `CONFLUENCE_URL` with your server URL
3. Optionally update `CONFLUENCE_ACCESS_TOKEN` with your own token

## Rollback Instructions
To revert to Confluence Cloud:
1. Change `cloud=False` to `cloud=True` in `create_confluence_client()`
2. Replace token authentication with username/password
3. Update URLs back to Atlassian Cloud format

## Token Information
The default token `NjgwODAxODIzMzIwOvsgoNEGsolZUPSWL7PT3TMvOv6m` is provided for testing purposes. For production use, generate your own Personal Access Token from your Confluence Server administration panel.
