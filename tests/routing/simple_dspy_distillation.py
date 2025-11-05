#!/usr/bin/env python3
"""
Simple DSPy Distillation - Working Implementation

This version avoids structured output issues by using simple string outputs
and manual parsing. Once this works, we can optimize the output format.
"""

import asyncio
import json
import os
import sys
import time
from typing import Any

import dspy

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from cogniverse_core.config.utils import get_config  # noqa: E402


class SimpleRoutingSignature(dspy.Signature):
    """Simple signature for query routing without structured outputs."""

    query = dspy.InputField(desc="User query to analyze")
    routing_decision = dspy.OutputField(
        desc="Routing decision as: video=true/false, text=true/false, temporal=pattern"
    )


class SimpleTeacher(dspy.Module):
    """Teacher model with chain-of-thought reasoning."""

    def __init__(self):
        super().__init__()
        self.route = dspy.ChainOfThought(SimpleRoutingSignature)

    def forward(self, query: str):
        return self.route(query=query)


class SimpleStudent(dspy.Module):
    """Student model without CoT for efficiency."""

    def __init__(self):
        super().__init__()
        self.route = dspy.Predict(SimpleRoutingSignature)

    def forward(self, query: str):
        return self.route(query=query)


class SimpleDistillation:
    """Simple distillation using string outputs."""

    def __init__(self):
        from cogniverse_core.config.manager import ConfigManager
        config_manager = ConfigManager()
        self.config = get_config(tenant_id="default", config_manager=config_manager)

        # Model selection
        self.teacher_model = "qwen2.5:7b"  # Larger, more capable
        self.student_model = "deepseek-r1:1.5b"  # Smaller, faster

        # Initialize modules
        self.teacher = SimpleTeacher()
        self.student = SimpleStudent()

    def setup_lm(self, model_name):
        """Setup LiteLLM wrapper for Ollama."""
        import litellm
        from dspy.clients.lm import BaseLM

        class SimpleLiteLLMWrapper(BaseLM):
            def __init__(self, model):
                self.model = model
                self.kwargs = {"temperature": 0.1, "max_tokens": 200}
                self.history = []

            def __call__(self, prompt=None, messages=None, **kwargs):
                try:
                    # Handle both prompt and messages format
                    if messages:
                        msgs = messages
                    elif prompt:
                        msgs = [{"role": "user", "content": prompt}]
                    else:
                        raise ValueError("Either prompt or messages must be provided")

                    print(f"   📝 LM calling: {str(msgs)[:50]}...")
                    response = litellm.completion(
                        model=self.model,
                        messages=msgs,
                        api_base="http://localhost:11434",
                        temperature=0.1,
                        max_tokens=200,
                    )
                    result = response.choices[0].message.content
                    print(f"   ✅ LM got: {result[:100]}...")
                    return result
                except Exception as e:
                    print(f"   ❌ LiteLLM call failed: {e}")
                    return "video=false, text=false, temporal=none"

        lm = SimpleLiteLLMWrapper(f"ollama/{model_name}")
        dspy.settings.configure(lm=lm)
        print(f"   ✅ Configured with {model_name}")

    def parse_routing_decision(self, decision_text: str) -> dict[str, Any]:
        """Parse routing decision from text output."""
        result = {
            "needs_video_search": False,
            "needs_text_search": False,
            "temporal_pattern": "none",
        }

        text_lower = decision_text.lower()

        # Check for video indicators
        if any(
            word in text_lower
            for word in [
                "video=true",
                "video: true",
                "video search",
                "visual content",
                "watch",
                "play",
            ]
        ):
            result["needs_video_search"] = True

        # Check for text indicators
        if any(
            word in text_lower
            for word in [
                "text=true",
                "text: true",
                "document",
                "paper",
                "article",
                "report",
            ]
        ):
            result["needs_text_search"] = True

        # Check for temporal patterns
        if "yesterday" in text_lower:
            result["temporal_pattern"] = "yesterday"
        elif "last_week" in text_lower or "last week" in text_lower:
            result["temporal_pattern"] = "last_week"
        elif "last_month" in text_lower or "last month" in text_lower:
            result["temporal_pattern"] = "last_month"
        elif "today" in text_lower:
            result["temporal_pattern"] = "today"

        return result

    async def generate_training_data(self) -> list[dspy.Example]:
        """Generate training data using teacher model."""

        queries = [
            # Clear video queries
            "Show me videos about deep learning",
            "Play the presentation from yesterday",
            "I want to watch the tutorial",
            "Find video recordings of the meeting",
            # Clear document queries
            "Find papers about transformers",
            "Get me the technical documentation",
            "Search for articles on neural networks",
            "I need the report from last week",
            # Mixed intent
            "Find all content about AI",
            "Search everything from yesterday",
            "Show me videos and documents",
            # Temporal queries
            "What happened yesterday",
            "Content from last month",
            "Today's updates",
            # Ambiguous (teacher will decide)
            "transformers architecture",
            "python tutorials",
            "machine learning",
            "latest research",
        ]

        print(f"\n📝 Generating training data from {len(queries)} queries...")

        self.setup_lm(self.teacher_model)
        training_examples = []

        for i, query in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] Query: '{query}'")

            try:
                # Get teacher's decision
                teacher_output = self.teacher(query)

                # Parse the routing decision
                routing_data = self.parse_routing_decision(
                    teacher_output.routing_decision
                )

                # Create training example
                example = dspy.Example(
                    query=query,
                    needs_video_search=routing_data["needs_video_search"],
                    needs_text_search=routing_data["needs_text_search"],
                    temporal_pattern=routing_data["temporal_pattern"],
                    routing_decision=teacher_output.routing_decision,
                ).with_inputs("query")

                training_examples.append(example)

                print(
                    f"   ✅ Video: {routing_data['needs_video_search']}, Text: {routing_data['needs_text_search']}, Temporal: {routing_data['temporal_pattern']}"
                )
                print(f"   📝 Decision: {teacher_output.routing_decision[:100]}...")

            except Exception as e:
                print(f"   ❌ Failed: {e}")

        return training_examples

    async def distill_knowledge(self, training_examples: list[dspy.Example]):
        """Distill teacher's knowledge into student model using BootstrapFewShot."""

        print(f"\n🔄 Distilling knowledge to {self.student_model}...")
        print(f"   Training examples: {len(training_examples)}")

        self.setup_lm(self.student_model)

        # Use BootstrapFewShot - simpler than MIPROv2 but still effective
        from dspy.teleprompt import BootstrapFewShot

        def routing_accuracy(example, prediction, trace=None):
            """Measure agreement with teacher (distillation metric)."""

            # Parse student prediction
            student_routing = self.parse_routing_decision(prediction.routing_decision)

            score = 0.0

            # Check routing accuracy against teacher labels
            if student_routing["needs_video_search"] == example.needs_video_search:
                score += 0.4
            if student_routing["needs_text_search"] == example.needs_text_search:
                score += 0.4
            if student_routing["temporal_pattern"] == example.temporal_pattern:
                score += 0.2

            return score

        # BootstrapFewShot optimizer
        optimizer = BootstrapFewShot(
            metric=routing_accuracy,
            max_bootstrapped_demos=4,  # Number of few-shot examples
            max_labeled_demos=8,  # Training examples to consider
            max_rounds=2,  # Optimization rounds
        )

        print("   🧠 Running BootstrapFewShot optimization...")

        start_time = time.time()

        # Split data for optimization
        train_size = int(0.8 * len(training_examples))
        train_set = training_examples[:train_size]
        val_set = training_examples[train_size:]

        if len(val_set) == 0:
            val_set = train_set[-2:]  # Use last 2 examples for validation

        print(f"   📊 Train: {len(train_set)}, Validation: {len(val_set)}")

        optimized_student = optimizer.compile(self.student, trainset=train_set)

        optimization_time = time.time() - start_time
        print(f"   ✅ Optimization complete in {optimization_time:.1f}s")

        # Test on validation set
        print("\n   📊 Validation Results:")
        val_correct = 0
        for example in val_set:
            try:
                pred = optimized_student(example.query)
                score = routing_accuracy(example, pred)
                if score >= 0.8:  # Consider >= 80% match as correct
                    val_correct += 1
            except Exception:
                pass

        val_accuracy = val_correct / len(val_set) if val_set else 0
        print(f"   📈 Validation accuracy: {val_accuracy:.1%}")

        return optimized_student


async def main():
    """Run the simple distillation process."""

    print("🚀 Simple DSPy Query Routing Distillation")
    print("=" * 60)

    distiller = SimpleDistillation()

    # Step 1: Generate training data with teacher
    training_data = await distiller.generate_training_data()

    if len(training_data) < 5:
        print("❌ Not enough training data generated")
        return

    # Step 2: Distill to student
    optimized_student = await distiller.distill_knowledge(training_data)

    # Step 3: Test the student
    print("\n📊 Testing distilled student model...")

    test_queries = [
        ("Show me yesterday's videos", True, False, "yesterday"),
        ("Find research papers", False, True, "none"),
        ("Search all content from last week", True, True, "last_week"),
    ]

    distiller.setup_lm(distiller.student_model)
    correct = 0

    for query, exp_video, exp_text, exp_temporal in test_queries:
        try:
            result = optimized_student(query)

            # Parse student prediction
            routing_data = distiller.parse_routing_decision(result.routing_decision)

            is_correct = (
                routing_data["needs_video_search"] == exp_video
                and routing_data["needs_text_search"] == exp_text
                and routing_data["temporal_pattern"] == exp_temporal
            )

            if is_correct:
                correct += 1

            print(f"\n🔍 Query: '{query}'")
            print(
                f"   Expected: video={exp_video}, text={exp_text}, temporal={exp_temporal}"
            )
            print(
                f"   Got: video={routing_data['needs_video_search']}, text={routing_data['needs_text_search']}, temporal={routing_data['temporal_pattern']}"
            )
            print(f"   {'✅ Correct' if is_correct else '❌ Wrong'}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    accuracy = correct / len(test_queries)
    print(f"\n📈 Student Accuracy: {accuracy:.1%}")

    # Save the model
    config = {
        "model": distiller.student_model,
        "accuracy": accuracy,
        "teacher_model": distiller.teacher_model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "optimizer": "BootstrapFewShot",
        "training_examples": len(training_data),
    }

    with open("simple_distilled_router.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n💾 Saved simple distilled model to simple_distilled_router.json")
    print("\n✅ Simple distillation complete!")
    print(f"   Student model: {distiller.student_model}")
    print("   Can now handle routing with string parsing!")


if __name__ == "__main__":
    asyncio.run(main())
