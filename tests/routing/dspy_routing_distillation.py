#!/usr/bin/env python3
"""
DSPy Model Distillation for Query Routing

Use a larger teacher model to generate high-quality routing decisions,
then distill this knowledge into a smaller student model (DeepSeek 1.5b or Gemma 1b).
"""

import dspy
import json
import asyncio
from typing import List, Dict, Tuple
import time
from datetime import datetime


class QueryRoutingSignature(dspy.Signature):
    """Signature for structured query routing."""

    query = dspy.InputField(desc="User query to analyze for routing")

    # Structured outputs
    needs_video_search = dspy.OutputField(
        desc="Boolean: true if query requires video/visual content search"
    )
    needs_text_search = dspy.OutputField(
        desc="Boolean: true if query requires text/document search"
    )
    temporal_pattern = dspy.OutputField(
        desc="Temporal pattern: 'yesterday', 'last_week', 'last_month', or 'none'"
    )
    confidence = dspy.OutputField(desc="Confidence score 0-1 for this routing decision")
    reasoning = dspy.OutputField(desc="Brief explanation of routing logic")


class TeacherRouter(dspy.Module):
    """Teacher model using a larger LLM for high-quality routing."""

    def __init__(self):
        super().__init__()
        # Use Chain of Thought for better reasoning
        self.route = dspy.ChainOfThought(QueryRoutingSignature)

    def forward(self, query: str):
        return self.route(query=query)


class StudentRouter(dspy.Module):
    """Student model to be trained via distillation."""

    def __init__(self):
        super().__init__()
        # Simple prediction without CoT for efficiency
        self.route = dspy.Predict(QueryRoutingSignature)

    def forward(self, query: str):
        return self.route(query=query)


class RoutingDistillation:
    """DSPy-based distillation framework for query routing."""

    def __init__(self, teacher_model="gpt-4", student_model="deepseek-r1:1.5b"):
        self.teacher_model = teacher_model
        self.student_model = student_model

        # Configure DSPy for teacher and student
        self.teacher_lm = None  # Will be configured
        self.student_lm = None  # Will be configured

        # Modules
        self.teacher = TeacherRouter()
        self.student = StudentRouter()

    def generate_training_data(self, queries: List[str]) -> List[dspy.Example]:
        """Use teacher model to generate high-quality training data."""
        print("🧑‍🏫 Generating training data with teacher model...")
        print(f"   Teacher: {self.teacher_model}")
        print(f"   Queries: {len(queries)}")

        # Configure DSPy with teacher model
        dspy.settings.configure(lm=self.teacher_lm)

        training_examples = []

        for i, query in enumerate(queries):
            try:
                # Get teacher's routing decision
                print(f"\n📝 Processing query {i+1}/{len(queries)}: '{query}'")

                teacher_output = self.teacher(query)

                # Create training example
                example = dspy.Example(
                    query=query,
                    needs_video_search=teacher_output.needs_video_search,
                    needs_text_search=teacher_output.needs_text_search,
                    temporal_pattern=teacher_output.temporal_pattern,
                    confidence=teacher_output.confidence,
                    reasoning=teacher_output.reasoning,
                ).with_inputs("query")

                training_examples.append(example)

                print(f"   ✅ Video: {teacher_output.needs_video_search}")
                print(f"   ✅ Text: {teacher_output.needs_text_search}")
                print(f"   ✅ Temporal: {teacher_output.temporal_pattern}")
                print(f"   ✅ Confidence: {teacher_output.confidence}")

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue

        print(f"\n✅ Generated {len(training_examples)} training examples")
        return training_examples

    def distill_to_student(
        self, training_examples: List[dspy.Example]
    ) -> StudentRouter:
        """Distill teacher's knowledge to student model using DSPy optimization."""
        print("\n🎓 Distilling to student model...")
        print(f"   Student: {self.student_model}")
        print(f"   Training examples: {len(training_examples)}")

        # Configure DSPy with student model
        dspy.settings.configure(lm=self.student_lm)

        # Use DSPy's BootstrapFewShotWithRandomSearch for distillation
        from dspy.teleprompt import BootstrapFewShotWithRandomSearch

        # Define metric for optimization
        def routing_metric(example, prediction, trace=None):
            """Measure how well student matches teacher."""
            score = 0.0

            # Exact match on routing decisions
            if prediction.needs_video_search == example.needs_video_search:
                score += 0.4
            if prediction.needs_text_search == example.needs_text_search:
                score += 0.4
            if prediction.temporal_pattern == example.temporal_pattern:
                score += 0.2

            return score

        # Optimize student model
        optimizer = BootstrapFewShotWithRandomSearch(
            metric=routing_metric,
            max_bootstrapped_demos=4,  # Few-shot examples
            max_labeled_demos=8,  # Training examples to use
            num_candidate_programs=10,  # Different prompt variations
            num_threads=1,
        )

        print("   🔄 Running optimization...")
        start_time = time.time()

        optimized_student = optimizer.compile(
            self.student,
            trainset=training_examples,
            valset=training_examples[-10:],  # Use last 10 as validation
        )

        optimization_time = time.time() - start_time
        print(f"   ✅ Optimization complete in {optimization_time:.1f}s")

        return optimized_student

    def evaluate_student(
        self, student: StudentRouter, test_queries: List[Tuple[str, Dict]]
    ):
        """Evaluate distilled student model."""
        print("\n📊 Evaluating student model...")

        # Configure DSPy with student model
        dspy.settings.configure(lm=self.student_lm)

        correct = 0
        total = 0

        for query, expected in test_queries:
            try:
                prediction = student(query)

                # Check accuracy
                routing_correct = (
                    prediction.needs_video_search == expected["video"]
                    and prediction.needs_text_search == expected["text"]
                )

                temporal_correct = prediction.temporal_pattern == expected.get(
                    "temporal", "none"
                )

                if routing_correct and temporal_correct:
                    correct += 1

                total += 1

                print(f"\n🔍 Query: '{query}'")
                print(
                    f"   Predicted: video={prediction.needs_video_search}, "
                    f"text={prediction.needs_text_search}, "
                    f"temporal={prediction.temporal_pattern}"
                )
                print(
                    f"   Expected: video={expected['video']}, "
                    f"text={expected['text']}, "
                    f"temporal={expected.get('temporal', 'none')}"
                )
                print(
                    f"   {'✅ Correct' if routing_correct and temporal_correct else '❌ Wrong'}"
                )

            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
                total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"\n📈 Student Accuracy: {accuracy:.1%} ({correct}/{total})")

        return accuracy


def create_comprehensive_query_set() -> List[str]:
    """Create diverse queries for teacher to label."""
    queries = [
        # Video queries
        "Show me videos about machine learning",
        "Play the presentation from yesterday",
        "I want to watch the demo",
        "Find video clips of robots",
        "Show me the recorded meeting",
        "Play tutorial videos on Python",
        # Document queries
        "Find documents about neural networks",
        "Show me the report from last week",
        "I need papers on deep learning",
        "Search for articles about AI",
        "Get me the documentation",
        "Find technical specifications",
        # Mixed queries
        "Find all content from yesterday",
        "Search videos and documents about AI",
        "Show me everything about transformers",
        "I need all materials from the workshop",
        # Temporal queries
        "What did we discuss yesterday",
        "Show me everything from last month",
        "Find content from this week",
        "Get me today's updates",
        # Ambiguous queries
        "Search for transformers",
        "Find Python content",
        "Show me the latest",
        "I need the presentation",
    ]

    return queries


async def main():
    """Run the distillation process."""
    print("🚀 DSPy Query Routing Distillation")
    print("=" * 60)

    # Initialize distillation framework
    distiller = RoutingDistillation(
        teacher_model="gpt-4",  # Or "claude-3" or another large model
        student_model="deepseek-r1:1.5b",  # Or "gemma:1b"
    )

    # For demo, we'll use mock LMs
    from dspy.utils.dummies import DummyLM

    # Configure teacher (in real use, this would be GPT-4 or Claude)
    distiller.teacher_lm = DummyLM(
        [
            {
                "needs_video_search": "true",
                "needs_text_search": "false",
                "temporal_pattern": "none",
                "confidence": "0.9",
                "reasoning": "Query asks to 'show videos' indicating video search",
            }
        ]
    )

    # Configure student (in real use, this would be DeepSeek 1.5b via Ollama)
    distiller.student_lm = DummyLM(
        [
            {
                "needs_video_search": "true",
                "needs_text_search": "false",
                "temporal_pattern": "none",
                "confidence": "0.8",
                "reasoning": "Video search needed",
            }
        ]
    )

    # Step 1: Generate training data with teacher
    queries = create_comprehensive_query_set()
    training_data = distiller.generate_training_data(queries[:20])

    # Step 2: Distill to student
    optimized_student = distiller.distill_to_student(training_data)

    # Step 3: Evaluate student
    test_queries = [
        (
            "Show me videos from yesterday",
            {"video": True, "text": False, "temporal": "yesterday"},
        ),
        ("Find documents about AI", {"video": False, "text": True, "temporal": "none"}),
        (
            "Search everything from last week",
            {"video": True, "text": True, "temporal": "last_week"},
        ),
    ]

    accuracy = distiller.evaluate_student(optimized_student, test_queries)

    # Save the distilled model
    print("\n💾 Saving distilled model...")
    with open("distilled_router.json", "w") as f:
        json.dump(
            {
                "student_model": distiller.student_model,
                "accuracy": accuracy,
                "timestamp": datetime.now().isoformat(),
                "training_examples": len(training_data),
            },
            f,
            indent=2,
        )

    print("\n✅ Distillation complete!")
    print(f"   Student model: {distiller.student_model}")
    print(f"   Accuracy: {accuracy:.1%}")
    print("   Ready for deployment!")


if __name__ == "__main__":
    asyncio.run(main())
