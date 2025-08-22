#!/usr/bin/env python3
"""
CORRECT DSPy MIPROv2 GLiNER Optimization

This shows what DSPy MIPROv2 actually optimizes:
- NOT the GLiNER labels
- BUT the logic for converting GLiNER entities to routing decisions
"""


import dspy


class GLiNERRoutingSignature(dspy.Signature):
    """What DSPy actually optimizes - the routing logic, not GLiNER config."""

    # Input: GLiNER entities as a string description
    entities = dspy.InputField(
        desc="List of entities found by GLiNER, e.g., 'video_content: videos (0.85), temporal_phrase: yesterday (0.92)'"
    )
    query = dspy.InputField(desc="Original user query")

    # Outputs: Routing decisions
    needs_video_search = dspy.OutputField(desc="True if query needs video search")
    needs_text_search = dspy.OutputField(desc="True if query needs text search")
    temporal_pattern = dspy.OutputField(desc="Temporal pattern or 'none'")
    reasoning = dspy.OutputField(desc="Brief explanation of routing decision")


class DSPyGLiNERRouter(dspy.Module):
    """This is what DSPy actually optimizes."""

    def __init__(self, gliner_labels: list[str], gliner_threshold: float):
        super().__init__()
        # Fixed GLiNER configuration - NOT optimized by DSPy
        self.gliner_labels = gliner_labels
        self.gliner_threshold = gliner_threshold

        # What DSPy optimizes: the routing logic
        self.route = dspy.ChainOfThought(GLiNERRoutingSignature)

    def forward(self, query: str, gliner_entities: list[dict]):
        """
        DSPy optimizes THIS logic - how to convert entities to routing.
        """
        # Format GLiNER entities for the prompt
        entities_str = ", ".join(
            [f"{e['label']}: {e['text']} ({e['score']:.2f})" for e in gliner_entities]
        )

        if not entities_str:
            entities_str = "No entities found"

        # DSPy optimizes this reasoning process
        routing = self.route(entities=entities_str, query=query)

        return routing


# Example of what MIPROv2 actually learns:

"""
MIPROv2 doesn't learn: "Use these GLiNER labels"

MIPROv2 DOES learn prompts like:

"Given GLiNER entities and a query, determine routing as follows:
- If entities contain 'video_content' or 'visual_content' with score > 0.5, set needs_video_search=True
- If entities contain 'document_content' or 'text_information', set needs_text_search=True
- If both video and text entities are present, route to both
- Extract temporal patterns from 'temporal_phrase' entities
- Consider query context even if entities are missing"

And few-shot examples like:
- Entities: "video_content: videos (0.85)", Query: "Show me videos" → video=True, text=False
- Entities: "temporal_phrase: yesterday (0.90)", Query: "from yesterday" → Use previous context
- Entities: "No entities found", Query: "search for reports" → text=True (learned from context)
"""


def demonstrate_real_miprov2():
    """Show what MIPROv2 actually optimizes."""

    print("🎯 What DSPy MIPROv2 ACTUALLY Optimizes:")
    print("=" * 60)

    print("\n❌ NOT This:")
    print("- Which GLiNER labels to use")
    print("- What threshold to set")
    print("- Which GLiNER model to load")

    print("\n✅ But This:")
    print("- How to interpret GLiNER entities")
    print("- When to trigger video vs text search based on entities")
    print("- How to handle missing entities")
    print("- How to combine multiple entities")
    print("- How to use query context when entities are ambiguous")

    print("\n📊 Example MIPROv2 Optimization Process:")
    print("\n1. Fixed GLiNER Config:")
    print('   Labels: ["video_content", "document_content", "temporal_phrase"]')
    print("   Threshold: 0.3")

    print("\n2. MIPROv2 Learns:")
    print("   - If 'video_content' entity → needs_video_search = True")
    print("   - If no entities but query has 'show me' → needs_video_search = True")
    print("   - If 'temporal_phrase' = 'yesterday' → temporal_pattern = 'yesterday'")
    print("   - If both video and document entities → route to both")

    print("\n3. The Optimized 'Program':")
    print("   MIPROv2 generates optimized prompts and few-shot examples")
    print("   that teach the LLM how to convert GLiNER output to routing")


if __name__ == "__main__":
    demonstrate_real_miprov2()

    print("\n\n💡 Key Insight:")
    print("DSPy MIPROv2 optimizes the REASONING about entities,")
    print("not the entity extraction itself!")
