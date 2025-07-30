from typing import Union
from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

class MCPRequest(BaseModel):
    jsonrpc: str
    method: str
    params: dict | None = None
    id: Union[str, int]

@app.post("/")
async def handle_mcp(request: Request, authorization: str = Header(default=None)):
    body = await request.json()
    mcp = MCPRequest(**body)

    if mcp.method == "describe":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": mcp.id,
            "result": {
                "name": "ChatGPT via MCP",
                "description": "Connects to OpenAI's GPT model via custom MCP wrapper.",
                "methods": ["complete"],
            }
        })

    if mcp.method == "parameters":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": mcp.id,
            "result": {
                "parameters": {
                    "prompt": {
                        "type": "string",
                        "description": "User input prompt"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system instruction for GPT"
                    }
                }
            }
        })

    if mcp.method == "complete":
        prompt = mcp.params.get("prompt", "")
        system = mcp.params.get("system_prompt", "You are a helpful assistant.")
        model = "gpt-4"

        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )

            result = completion.choices[0].message["content"]
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": mcp.id,
                "result": {
                    "completion": result,
                }
            })

        except Exception as e:
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": mcp.id,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            })

    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id": mcp.id,
        "error": {
            "code": -32601,
            "message": f"Method '{mcp.method}' not found"
        }
    })

@app.get("/.well-known/oauth-authorization-server")
async def oauth_metadata():
    return JSONResponse(content={"issuer": "https://your-server-url"})

@app.post("/register")
async def register_stub():
    return JSONResponse(content={"status": "ok"})
