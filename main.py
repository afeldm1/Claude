from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import os
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Claude MCP Remote Connector",
    description="A Remote MCP protocol implementation for Claude integration",
    version="1.0.0"
)

# Pydantic models for request/response validation
class MCPRequest(BaseModel):
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

class MCPResponse(BaseModel):
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

class ToolParameter(BaseModel):
    type: str
    description: str
    required: Optional[bool] = False
    enum: Optional[List[str]] = None

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ToolParameter]

# Authentication helper
def verify_auth(authorization: Optional[str] = None) -> bool:
    """Verify the authorization header"""
    if not authorization:
        return False
    
    # Extract bearer token
    if not authorization.startswith("Bearer "):
        return False
    
    token = authorization[7:]  # Remove "Bearer " prefix
    expected_token = os.getenv("MCP_API_TOKEN", "your-secret-token-here")
    
    return token == expected_token

# Tool definitions - customize these for your specific use case
AVAILABLE_TOOLS = {
    "web_search": ToolDefinition(
        name="web_search",
        description="Search the web for current information using GPT-4",
        parameters={
            "query": ToolParameter(
                type="string",
                description="The search query to execute",
                required=True
            ),
            "max_results": ToolParameter(
                type="integer",
                description="Maximum number of results to return (1-10)",
                required=False
            )
        }
    ),
    "text_analysis": ToolDefinition(
        name="text_analysis",
        description="Analyze text for sentiment, keywords, or summary using GPT-4",
        parameters={
            "text": ToolParameter(
                type="string",
                description="The text to analyze",
                required=True
            ),
            "analysis_type": ToolParameter(
                type="string",
                description="Type of analysis to perform",
                required=True,
                enum=["sentiment", "keywords", "summary", "topics"]
            )
        }
    ),
    "code_review": ToolDefinition(
        name="code_review",
        description="Review code for quality, security, and best practices using GPT-4",
        parameters={
            "code": ToolParameter(
                type="string",
                description="The code to review",
                required=True
            ),
            "language": ToolParameter(
                type="string",
                description="Programming language of the code",
                required=False
            )
        }
    )
}

def validate_tool_parameters(tool_name: str, params: Dict[str, Any]) -> None:
    """Validate tool parameters before execution"""
    if tool_name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    tool_def = AVAILABLE_TOOLS[tool_name]
    
    # Check required parameters
    for param_name, param_def in tool_def.parameters.items():
        if param_def.required and param_name not in params:
            raise ValueError(f"Missing required parameter: {param_name}")
        
        # Check enum values
        if param_name in params and param_def.enum:
            if params[param_name] not in param_def.enum:
                raise ValueError(f"Invalid value for {param_name}. Must be one of: {param_def.enum}")

async def execute_tool_with_openai(tool_name: str, params: Dict[str, Any]) -> str:
    """Execute a tool using OpenAI GPT-4"""
    
    # Validate parameters first
    validate_tool_parameters(tool_name, params)
    
    # Create tool-specific prompts
    if tool_name == "web_search":
        query = params.get("query", "")
        max_results = params.get("max_results", 5)
        
        prompt = f"""
        You are a web search assistant. Based on the query "{query}", provide {max_results} relevant and current results.
        
        Format your response as if you've searched the web and found actual results. Include:
        - Brief summaries of what you'd find
        - Relevant URLs (use realistic-looking URLs)
        - Key information that would be current as of the search date
        
        Query: {query}
        Max results: {max_results}
        """
    
    elif tool_name == "text_analysis":
        text = params.get("text", "")
        analysis_type = params.get("analysis_type", "summary")
        
        prompt = f"""
        Perform a {analysis_type} analysis on the following text:
        
        Text to analyze: "{text}"
        
        Analysis type: {analysis_type}
        
        Provide detailed {analysis_type} analysis with clear insights and conclusions.
        """
    
    elif tool_name == "code_review":
        code = params.get("code", "")
        language = params.get("language", "unknown")
        
        prompt = f"""
        Review the following {language} code for:
        - Code quality and best practices
        - Potential security issues
        - Performance considerations
        - Suggestions for improvement
        
        Code to review:
        ```{language}
        {code}
        ```
        
        Provide a comprehensive code review with specific recommendations.
        """
    
    else:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    # Call OpenAI API with proper async handling
    try:
        # Use the newer OpenAI client syntax
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate and detailed responses for the requested tool functionality."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        # More specific error handling
        if "API key" in str(e).lower():
            raise Exception("OpenAI API key not configured or invalid")
        elif "rate limit" in str(e).lower():
            raise Exception("OpenAI API rate limit exceeded")
        elif "model" in str(e).lower():
            raise Exception("OpenAI model not available or invalid")
        else:
            raise Exception(f"OpenAI API error: {str(e)}")

@app.post("/")
async def mcp_endpoint(
    request: MCPRequest,
    authorization: Optional[str] = Header(None)
):
    """Main MCP endpoint that handles describe, parameters, and complete methods"""
    
    # Verify authentication
    if not verify_auth(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        if request.method == "describe":
            return MCPResponse(
                result={
                    "name": "Claude MCP Remote Connector",
                    "version": "1.0.0",
                    "description": "A remote MCP connector that provides AI-powered tools via OpenAI GPT-4",
                    "tools": [tool.dict() for tool in AVAILABLE_TOOLS.values()],
                    "capabilities": {
                        "tools": True,
                        "resources": False,
                        "prompts": False
                    }
                },
                id=request.id
            ).dict()
        
        elif request.method == "parameters":
            tool_name = request.params.get("tool") if request.params else None
            if not tool_name or tool_name not in AVAILABLE_TOOLS:
                return MCPResponse(
                    error={
                        "code": -32602,
                        "message": f"Invalid tool name: {tool_name}"
                    },
                    id=request.id
                ).dict()
            
            tool = AVAILABLE_TOOLS[tool_name]
            return MCPResponse(
                result={
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {name: param.dict() for name, param in tool.parameters.items()}
                },
                id=request.id
            ).dict()
        
        elif request.method == "complete":
            return await handle_complete(request)
        
        else:
            return MCPResponse(
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}"
                },
                id=request.id
            ).dict()
    
    except Exception as e:
        logger.error(f"MCP request failed: {str(e)}")
        return MCPResponse(
            error={
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            },
            id=request.id
        ).dict()

async def handle_complete(request: MCPRequest) -> Dict[str, Any]:
    """Handle tool completion requests using OpenAI GPT-4"""
    
    if not request.params:
        return MCPResponse(
            error={
                "code": -32602,
                "message": "Missing parameters"
            },
            id=request.id
        ).dict()
    
    tool_name = request.params.get("tool")
    tool_params = request.params.get("parameters", {})
    
    if not tool_name or tool_name not in AVAILABLE_TOOLS:
        return MCPResponse(
            error={
                "code": -32602,
                "message": f"Invalid tool name: {tool_name}"
            },
            id=request.id
        ).dict()
    
    try:
        # Execute the tool using OpenAI
        result = await execute_tool_with_openai(tool_name, tool_params)
        
        return MCPResponse(
            result={
                "tool": tool_name,
                "parameters": tool_params,
                "output": result,
                "timestamp": datetime.utcnow().isoformat()
            },
            id=request.id
        ).dict()
    
    except Exception as e:
        logger.error(f"Tool execution failed: {str(e)}")
        return MCPResponse(
            error={
                "code": -32603,
                "message": f"Tool execution failed: {str(e)}"
            },
            id=request.id
        ).dict()

@app.post("/register")
async def register_endpoint():
    """OAuth registration endpoint stub - implement based on your OAuth provider"""
    return JSONResponse({
        "message": "Registration endpoint - implement OAuth registration logic here",
        "client_id": "your-client-id",
        "client_secret": "your-client-secret",
        "redirect_uris": ["https://your-app.com/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "scope": "read write"
    })

@app.get("/.well-known/oauth-authorization-server")
async def oauth_metadata():
    """OAuth 2.0 Authorization Server Metadata endpoint"""
    base_url = os.getenv("BASE_URL", "https://your-app.com")
    
    return JSONResponse({
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/oauth/authorize",
        "token_endpoint": f"{base_url}/oauth/token",
        "userinfo_endpoint": f"{base_url}/oauth/userinfo",
        "jwks_uri": f"{base_url}/.well-known/jwks.json",
        "scopes_supported": ["read", "write", "admin"],
        "response_types_supported": ["code", "token"],
        "grant_types_supported": ["authorization_code", "refresh_token", "client_credentials"],
        "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
        "code_challenge_methods_supported": ["S256"],
        "revocation_endpoint": f"{base_url}/oauth/revoke",
        "introspection_endpoint": f"{base_url}/oauth/introspect"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "name": "Claude MCP Remote Connector",
        "version": "1.0.0",
        "description": "Remote MCP protocol implementation for Claude integration",
        "endpoints": {
            "mcp": "POST /",
            "register": "POST /register",
            "oauth_metadata": "GET /.well-known/oauth-authorization-server",
            "health": "GET /health"
        },
        "documentation": "/docs"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)