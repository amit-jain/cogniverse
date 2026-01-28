"""
Pytest configuration for synthetic data tests

Sets up DSPy with dummy LM for testing
"""

import dspy
import pytest


class DummyLM(dspy.LM):
    """Dummy LM that returns predictable outputs for testing"""

    def __init__(self):
        super().__init__(model="dummy")
        self.call_count = 0
        self.history = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        """Return a dummy response"""
        self.call_count += 1

        # Parse prompt to determine response type
        if messages and len(messages) > 0:
            prompt_text = str(messages[-1].get("content", ""))
        elif prompt:
            prompt_text = str(prompt)
        else:
            prompt_text = ""

        # Store in history
        self.history.append({"prompt": prompt_text, "messages": messages})

        # Determine response based on prompt
        # Extract entities from prompt to ensure query contains them
        response_text = None
        if "modality" in prompt_text.lower() or (
            "topics" in prompt_text.lower() and "entities" not in prompt_text.lower()
        ):
            response_text = '{"query": "find machine learning tutorial video"}'
        elif "entities" in prompt_text.lower() or "entity_types" in prompt_text.lower():
            # Extract the first entity from the prompt to ensure it appears in the query
            entity_name = "TensorFlow"  # default

            # Try to extract actual entities from prompt
            for possible_entity in [
                "TensorFlow",
                "PyTorch",
                "Python",
                "Networks",
                "Tutorial",
                "Learn",
            ]:
                if possible_entity in prompt_text:
                    entity_name = possible_entity
                    break

            # Generate query that explicitly contains the entity, with reasoning for ChainOfThought
            response_text = f'{{"reasoning": "Including {entity_name} as the primary entity since it is a key technology", "query": "find {entity_name} machine learning tutorial"}}'
        elif "agent" in prompt_text.lower():
            response_text = (
                '{"agent_name": "video_search_agent", "reasoning": "Video content"}'
            )
        else:
            response_text = '{"query": "test query"}'

        # Return in format DSPy expects
        return [response_text]

    def inspect_history(self, n=1):
        """Return recent history for testing"""
        return self.history[-n:] if self.history else []


@pytest.fixture(scope="session", autouse=True)
def setup_dspy():
    """Set up DSPy with dummy LM for all tests in synthetic directory"""
    dummy_lm = DummyLM()
    dspy.configure(lm=dummy_lm)
    yield dummy_lm
    # Cleanup after ALL synthetic tests complete
    dspy.configure(lm=None)


@pytest.fixture(autouse=True)
def reset_dspy_lm(setup_dspy):
    """Reset the dummy LM state between tests"""
    # Reset state and reconfigure for this test
    setup_dspy.call_count = 0
    setup_dspy.history = []
    dspy.configure(lm=setup_dspy)

    yield setup_dspy

    # Don't restore original - let session fixture handle final cleanup
