# src/tools/a2a_utils.py
import httpx
import json
import uuid
from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field

# --- A2A Protocol Data Models ---
class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str

class DataPart(BaseModel):
    type: Literal["data"] = "data"
    data: Dict[str, Any]

class FilePart(BaseModel):
    type: Literal["file"] = "file"
    file_uri: str
    mime_type: str

class A2AMessage(BaseModel):
    role: str
    parts: List[Union[TextPart, DataPart, FilePart]]

class Task(BaseModel):
    id: str
    messages: List[A2AMessage]

class AgentCard(BaseModel):
    name: str
    description: str
    url: str
    version: str
    protocol: str
    protocol_version: str
    capabilities: List[str]
    skills: List[Dict[str, Any]]

# --- A2A Client Helper Functions ---
class A2AClient:
    """A helper class for making A2A protocol requests to agents."""
    
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
    
    async def send_task(self, agent_url: str, query: str, **kwargs) -> Dict[str, Any]:
        """
        Send a task to an A2A-compliant agent.
        
        Args:
            agent_url: The base URL of the agent
            query: The query string to send
            **kwargs: Additional parameters (top_k, start_date, end_date, etc.)
        
        Returns:
            Dict containing the agent's response
        """
        task_id = str(uuid.uuid4())
        task_payload = {
            "id": task_id,
            "messages": [{
                "role": "user",
                "parts": [{
                    "type": "data",
                    "data": {"query": query, **kwargs}
                }]
            }]
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{agent_url}/tasks/send",
                    json=task_payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            return {"error": f"Request failed: {e}"}
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error {e.response.status_code}: {e.response.text}"}
    
    async def get_agent_card(self, agent_url: str) -> Dict[str, Any]:
        """
        Retrieve the agent card from an A2A-compliant agent.
        
        Args:
            agent_url: The base URL of the agent
        
        Returns:
            Dict containing the agent card information
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{agent_url}/agent.json")
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            return {"error": f"Request failed: {e}"}
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error {e.response.status_code}: {e.response.text}"}

# --- Utility Functions ---
def create_text_message(text: str, role: str = "user") -> A2AMessage:
    """Create a text message in A2A format."""
    return A2AMessage(
        role=role,
        parts=[TextPart(text=text)]
    )

def create_data_message(data: Dict[str, Any], role: str = "user") -> A2AMessage:
    """Create a data message in A2A format."""
    return A2AMessage(
        role=role,
        parts=[DataPart(data=data)]
    )

def create_task(messages: List[A2AMessage], task_id: Optional[str] = None) -> Task:
    """Create a task in A2A format."""
    if task_id is None:
        task_id = str(uuid.uuid4())
    
    return Task(id=task_id, messages=messages)

def extract_data_from_message(message: A2AMessage) -> Optional[Dict[str, Any]]:
    """Extract data from the first DataPart in a message."""
    for part in message.parts:
        if isinstance(part, DataPart):
            return part.data
    return None

def extract_text_from_message(message: A2AMessage) -> Optional[str]:
    """Extract text from the first TextPart in a message."""
    for part in message.parts:
        if isinstance(part, TextPart):
            return part.text
    return None

# --- Agent Discovery Functions ---
async def discover_agents(agent_urls: List[str]) -> Dict[str, AgentCard]:
    """
    Discover agent capabilities by fetching their agent cards.
    
    Args:
        agent_urls: List of agent URLs to query
    
    Returns:
        Dict mapping agent names to their agent cards
    """
    client = A2AClient()
    discovered_agents = {}
    
    for url in agent_urls:
        try:
            card_data = await client.get_agent_card(url)
            if "error" not in card_data:
                agent_card = AgentCard(**card_data)
                discovered_agents[agent_card.name] = agent_card
        except Exception as e:
            print(f"Failed to discover agent at {url}: {e}")
    
    return discovered_agents

def format_search_results(results: List[Dict[str, Any]], result_type: str = "generic") -> str:
    """
    Format search results for display or further processing.
    
    Args:
        results: List of search result dictionaries
        result_type: Type of results ("text", "video", "generic")
    
    Returns:
        Formatted string representation of results
    """
    if not results:
        return "No results found."
    
    formatted_results = []
    
    if result_type == "text":
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. Document ID: {result.get('doc_id', 'N/A')}\n"
                f"   Content: {result.get('content', 'N/A')[:200]}...\n"
                f"   Score: {result.get('score', 'N/A')}\n"
            )
    elif result_type == "video":
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. Video ID: {result.get('video_id', 'N/A')}\n"
                f"   Frame ID: {result.get('frame_id', 'N/A')}\n"
                f"   Time: {result.get('start_time', 'N/A')}s - {result.get('end_time', 'N/A')}s\n"
                f"   Relevance: {result.get('relevance', 'N/A')}\n"
            )
    else:
        for i, result in enumerate(results, 1):
            formatted_results.append(f"{i}. {json.dumps(result, indent=2)}\n")
    
    return "\n".join(formatted_results) 