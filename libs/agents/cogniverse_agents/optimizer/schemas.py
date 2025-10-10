"""
Schema definitions for the Agentic Router

These schemas implement the NEW_PROPOSAL.md design:
- RoutingDecision with search_modality and generation_type
- AgenticRouter signature for DSPy
"""

import dspy
from pydantic import BaseModel, Field


class RoutingDecision(BaseModel):
    """Defines the strict schema for the router's JSON output."""
    search_modality: str = Field(
        description="Must be either 'video' or 'text'.",
        pattern="^(video|text)$"
    )
    generation_type: str = Field(
        description="Must be 'detailed_report', 'summary', or 'raw_results'.",
        pattern="^(detailed_report|summary|raw_results)$"
    )


class AgenticRouter(dspy.Signature):
    """
    You are a precise and efficient routing agent. Your sole responsibility is to analyze 
    the user's query and any provided conversation history. Based on this analysis, you 
    MUST generate a single, valid JSON object that dictates the next steps. The JSON 
    object must conform to the provided schema. Do not add any conversational text or 
    explanations.
    """
    
    conversation_history: str = dspy.InputField(
        desc="Recent turns in the conversation. Can be empty.",
        default=""
    )
    user_query: str = dspy.InputField(
        desc="The latest query from the user."
    )
    
    routing_decision: RoutingDecision = dspy.OutputField(
        desc="A single valid JSON object containing the routing decision."
    )