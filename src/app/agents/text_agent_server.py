# src/agents/text_agent_server.py
import os
from typing import Any, Dict, List, Union

import uvicorn
from elasticsearch import Elasticsearch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# --- A2A Protocol Data Models ---
class TextPart(BaseModel):
    type: str = Field("text", const=True)
    text: str


class DataPart(BaseModel):
    type: str = Field("data", const=True)
    data: Dict[str, Any]


class A2AMessage(BaseModel):
    role: str
    parts: List[Union[TextPart, DataPart]] = Field(..., discriminator="type")


class Task(BaseModel):
    id: str
    messages: List[A2AMessage]


# --- Text Search Agent Implementation ---
class TextSearchAgent:
    def __init__(self):
        print("Initializing TextSearchAgent...")
        try:
            self.client = Elasticsearch(
                cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
                api_key=os.getenv("ELASTIC_API_KEY"),
            )
            print("Successfully connected to Elasticsearch.")
        except Exception as e:
            print(f"Could not connect to Elasticsearch. Error: {e}")
            self.client = None

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("SentenceTransformer model loaded.")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.client:
            raise ConnectionError("Elasticsearch connection is not available.")

        query_vector = self.model.encode(query).tolist()

        search_body = {
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {"standard": {"query": {"match": {"content": query}}}},
                        {
                            "knn": {
                                "field": "content_vector",
                                "query_vector": query_vector,
                                "k": 50,
                                "num_candidates": 100,
                            }
                        },
                    ],
                    "rank_constant": 20,
                    "window_size": 100,
                }
            },
            "size": top_k,
            "_source": ["doc_id", "content"],
        }

        try:
            response = self.client.search(index="your-text-index", body=search_body)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            print(f"An error occurred during Elasticsearch query: {e}")
            return []


# --- FastAPI Server ---
app = FastAPI(title="Text Search Agent")
text_agent = TextSearchAgent()


@app.get("/agent.json", summary="Get Agent Card")
async def get_agent_card():
    return {
        "name": "TextSearchAgent",
        "description": "Searches for information in text documents and internal reports.",
        "url": "/tasks/send",
        "version": "1.0",
        "protocol": "a2a",
        "protocol_version": "0.2.1",
        "capabilities": ["tasks/send"],
        "skills": [
            {
                "name": "hybridTextSearch",
                "description": "Performs hybrid search combining lexical and semantic search over text documents",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            }
        ],
    }


@app.post("/tasks/send", status_code=202, summary="Send a Search Task")
async def send_task(task: Task):
    if not task.messages:
        raise HTTPException(status_code=400, detail="Task contains no messages.")

    last_message = task.messages[-1]
    data_part = next(
        (part for part in last_message.parts if isinstance(part, DataPart)), None
    )

    if not data_part:
        raise HTTPException(status_code=400, detail="No DataPart found in the message.")

    query_data = data_part.data
    query = query_data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in DataPart.")

    try:
        search_results = text_agent.search(
            query=query, top_k=query_data.get("top_k", 10)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    return {"task_id": task.id, "status": "completed", "results": search_results}


if __name__ == "__main__":
    print(
        "--- To run the Text Search Agent server, use: uvicorn text_agent_server:app --reload --port 8002 ---"
    )
    uvicorn.run(app, host="0.0.0.0", port=8002)
