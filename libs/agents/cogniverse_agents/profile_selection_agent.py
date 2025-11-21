"""
ProfileSelectionAgent - Autonomous A2A agent for selecting optimal backend profiles.

Uses LLM-based reasoning (SmolLM) to analyze queries and select the most appropriate
backend profile based on query characteristics, modality, and complexity.
"""

import logging
from typing import Any, Dict, List, Optional

import dspy
from pydantic import BaseModel, Field

from cogniverse_core.agents.dspy_a2a_base import DSPyA2AAgentBase

logger = logging.getLogger(__name__)


class ProfileCandidate(BaseModel):
    """Candidate profile with score"""
    profile_name: str
    score: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(description="Why this profile was selected")


class ProfileSelectionResult(BaseModel):
    """Result of profile selection"""
    query: str
    selected_profile: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    query_intent: str = Field(description="Detected query intent")
    modality: str = Field(description="Target modality")
    complexity: str = Field(description="Query complexity: simple, medium, complex")
    alternatives: List[ProfileCandidate] = Field(default_factory=list, description="Alternative profiles")


class ProfileSelectionSignature(dspy.Signature):
    """Select optimal backend profile based on query analysis"""
    query: str = dspy.InputField(desc="User query to analyze")
    available_profiles: str = dspy.InputField(desc="Comma-separated list of available profiles")

    selected_profile: str = dspy.OutputField(desc="Best matching profile name")
    confidence: str = dspy.OutputField(desc="Confidence score 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="Explanation for profile selection")
    query_intent: str = dspy.OutputField(desc="Detected intent: text_search, video_search, image_search, etc.")
    modality: str = dspy.OutputField(desc="Target modality: video, image, text, audio")
    complexity: str = dspy.OutputField(desc="Query complexity: simple, medium, complex")


class ProfileSelectionModule(dspy.Module):
    """DSPy module for profile selection with LLM reasoning"""

    def __init__(self):
        super().__init__()
        # Use ChainOfThought for reasoning
        self.selector = dspy.ChainOfThought(ProfileSelectionSignature)

    def forward(self, query: str, available_profiles: str) -> dspy.Prediction:
        """Select optimal profile using LLM reasoning"""
        try:
            return self.selector(query=query, available_profiles=available_profiles)
        except Exception as e:
            logger.warning(f"Profile selection failed: {e}, using fallback")
            # Fallback: simple heuristic
            return self._fallback_selection(query, available_profiles)

    def _fallback_selection(self, query: str, available_profiles: str) -> dspy.Prediction:
        """Fallback profile selection using heuristics"""
        profiles = [p.strip() for p in available_profiles.split(",")]
        query_lower = query.lower()

        # Simple heuristic: check for keywords
        if "video" in query_lower:
            modality = "video"
            intent = "video_search"
        elif "image" in query_lower or "picture" in query_lower or "photo" in query_lower:
            modality = "image"
            intent = "image_search"
        else:
            modality = "video"  # Default
            intent = "text_search"

        # Select first matching profile or default
        selected = profiles[0] if profiles else "default"

        # Determine complexity
        word_count = len(query.split())
        if word_count <= 3:
            complexity = "simple"
        elif word_count <= 10:
            complexity = "medium"
        else:
            complexity = "complex"

        return dspy.Prediction(
            selected_profile=selected,
            confidence="0.5",
            reasoning=f"Fallback selection based on {modality} modality detection",
            query_intent=intent,
            modality=modality,
            complexity=complexity
        )


class ProfileSelectionAgent(DSPyA2AAgentBase):
    """
    Autonomous A2A agent for backend profile selection.

    Uses LLM-based reasoning (SmolLM or similar small model) to analyze queries
    and select the optimal backend profile based on:
    - Query intent (text/video/image search)
    - Target modality
    - Query complexity
    - Profile capabilities

    Capabilities:
    - LLM-based query analysis
    - Profile matching and ranking
    - Reasoning explanation
    - Alternative profile suggestions
    """

    def __init__(
        self,
        tenant_id: str = "default",
        available_profiles: Optional[List[str]] = None,
        port: int = 8011
    ):
        """
        Initialize ProfileSelectionAgent

        Args:
            tenant_id: Tenant identifier
            available_profiles: List of available backend profiles
            port: Port for A2A server
        """
        self.tenant_id = tenant_id
        self.available_profiles = available_profiles or [
            "video_colpali_base",
            "video_colpali_large",
            "video_videoprism_base",
            "image_colpali_base",
            "text_bge_base"
        ]

        # Initialize DSPy module
        selection_module = ProfileSelectionModule()

        # Initialize base class
        super().__init__(
            agent_name="profile_selection_agent",
            agent_description="Autonomous profile selection with LLM-based reasoning",
            dspy_module=selection_module,
            capabilities=[
                "profile_selection",
                "query_analysis",
                "modality_detection",
                "intent_classification",
                "profile_ranking"
            ],
            port=port,
            version="1.0.0"
        )

        logger.info(
            f"ProfileSelectionAgent initialized for tenant: {tenant_id}, "
            f"profiles: {len(self.available_profiles)}"
        )

    async def _process(self, dspy_input: Dict[str, Any]) -> ProfileSelectionResult:
        """
        Process profile selection request

        Args:
            dspy_input: Input with 'query' and optional 'available_profiles'

        Returns:
            ProfileSelectionResult with selected profile and reasoning
        """
        query = dspy_input.get("query", "")
        profiles = dspy_input.get("available_profiles", self.available_profiles)

        if not query:
            return ProfileSelectionResult(
                query="",
                selected_profile=self.available_profiles[0] if self.available_profiles else "default",
                confidence=0.0,
                reasoning="Empty query, using default profile",
                query_intent="unknown",
                modality="video",
                complexity="simple",
                alternatives=[]
            )

        # Convert profiles list to comma-separated string for DSPy
        profiles_str = ", ".join(profiles) if isinstance(profiles, list) else profiles

        # Select profile using DSPy LLM reasoning
        result = self.dspy_module.forward(query=query, available_profiles=profiles_str)

        # Parse confidence
        try:
            confidence = float(result.confidence)
        except (ValueError, AttributeError):
            confidence = 0.5

        # Generate alternative profiles (top 3)
        alternatives = self._generate_alternatives(
            query, profiles, result.selected_profile, result.modality
        )

        return ProfileSelectionResult(
            query=query,
            selected_profile=result.selected_profile,
            confidence=confidence,
            reasoning=result.reasoning,
            query_intent=result.query_intent,
            modality=result.modality,
            complexity=result.complexity,
            alternatives=alternatives
        )

    def _generate_alternatives(
        self,
        query: str,
        profiles: List[str],
        selected: str,
        modality: str
    ) -> List[ProfileCandidate]:
        """Generate alternative profile suggestions"""
        alternatives = []

        if isinstance(profiles, str):
            profiles = [p.strip() for p in profiles.split(",")]

        # Score profiles based on modality match
        for profile in profiles:
            if profile == selected:
                continue

            # Simple scoring: check if profile name matches modality
            score = 0.3  # Base score
            if modality.lower() in profile.lower():
                score += 0.4

            if score > 0.3:  # Only include relevant alternatives
                alternatives.append(ProfileCandidate(
                    profile_name=profile,
                    score=score,
                    reasoning=f"Alternative profile for {modality} modality"
                ))

        # Sort by score and return top 3
        alternatives.sort(key=lambda x: x.score, reverse=True)
        return alternatives[:3]

    def _dspy_to_a2a_output(self, dspy_output: Any) -> Dict[str, Any]:
        """Convert ProfileSelectionResult to A2A format"""
        if isinstance(dspy_output, ProfileSelectionResult):
            return {
                "status": "success",
                "agent": self.agent_name,
                "query": dspy_output.query,
                "selected_profile": dspy_output.selected_profile,
                "confidence": dspy_output.confidence,
                "reasoning": dspy_output.reasoning,
                "query_intent": dspy_output.query_intent,
                "modality": dspy_output.modality,
                "complexity": dspy_output.complexity,
                "alternatives": [a.model_dump() for a in dspy_output.alternatives]
            }
        else:
            return {
                "status": "success",
                "agent": self.agent_name,
                "output": str(dspy_output)
            }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Define agent skills for A2A protocol"""
        return [
            {
                "name": "select_profile",
                "description": "Select optimal backend profile using LLM reasoning",
                "input_schema": {
                    "query": "string",
                    "available_profiles": "optional array of strings"
                },
                "output_schema": {
                    "selected_profile": "string",
                    "confidence": "float",
                    "reasoning": "string",
                    "query_intent": "string",
                    "modality": "string",
                    "complexity": "string",
                    "alternatives": "array of {profile_name, score, reasoning}"
                },
                "examples": [
                    {
                        "input": {
                            "query": "Show me videos about machine learning",
                            "available_profiles": ["video_colpali_base", "text_bge_base"]
                        },
                        "output": {
                            "selected_profile": "video_colpali_base",
                            "confidence": 0.9,
                            "reasoning": "Query requests video content, best matched by video_colpali_base",
                            "query_intent": "video_search",
                            "modality": "video",
                            "complexity": "simple"
                        }
                    }
                ]
            }
        ]


# FastAPI app for standalone deployment
from fastapi import FastAPI

app = FastAPI(
    title="ProfileSelectionAgent",
    description="Autonomous profile selection agent with LLM reasoning",
    version="1.0.0"
)

# Global agent instance
profile_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global profile_agent

    import os
    tenant_id = os.getenv("TENANT_ID", "default")
    profile_agent = ProfileSelectionAgent(tenant_id=tenant_id)
    logger.info("ProfileSelectionAgent started")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not profile_agent:
        return {"status": "initializing"}
    return profile_agent.app.routes[2].endpoint()


@app.get("/agent.json")
async def agent_card():
    """Agent card endpoint"""
    if not profile_agent:
        return {"error": "Agent not initialized"}
    return profile_agent.app.routes[0].endpoint()


@app.post("/tasks/send")
async def process_task(task: Dict[str, Any]):
    """Process A2A task"""
    if not profile_agent:
        return {"error": "Agent not initialized"}
    return await profile_agent.app.routes[1].endpoint(task)


if __name__ == "__main__":
    import uvicorn
    agent = ProfileSelectionAgent(tenant_id="default", port=8011)
    logger.info("Starting ProfileSelectionAgent on port 8011...")
    agent.run()
