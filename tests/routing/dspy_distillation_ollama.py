#!/usr/bin/env python3
"""
DSPy Distillation with Ollama Models

Practical implementation using available Ollama models for distillation.
Teacher: Larger model (e.g., DeepSeek R1 7B, Qwen 7B)
Student: Smaller model (e.g., DeepSeek R1 1.5B, Gemma 2B)
"""

import dspy
import json
import asyncio
import time
from typing import List, Dict, Any, Tuple
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.common.config import get_config

class StructuredRoutingSignature(dspy.Signature):
    """Structured output signature for query routing."""
    
    query = dspy.InputField(desc="User query to analyze")
    
    # Structured boolean outputs
    needs_video_search = dspy.OutputField(desc="true if query needs video search, false otherwise")
    needs_text_search = dspy.OutputField(desc="true if query needs document/text search, false otherwise") 
    temporal_pattern = dspy.OutputField(desc="Temporal pattern: yesterday, today, last_week, last_month, or none")
    reasoning = dspy.OutputField(desc="One sentence explanation")

class QueryRoutingTeacher(dspy.Module):
    """Teacher model with chain-of-thought reasoning."""
    
    def __init__(self):
        super().__init__()
        self.route = dspy.ChainOfThought(StructuredRoutingSignature)
    
    def forward(self, query: str):
        return self.route(query=query)

class QueryRoutingStudent(dspy.Module):
    """Efficient student model without CoT."""
    
    def __init__(self):
        super().__init__()
        self.route = dspy.Predict(StructuredRoutingSignature)
    
    def forward(self, query: str):
        return self.route(query=query)

class OllamaDistillation:
    """Distillation using Ollama models."""
    
    def __init__(self):
        self.config = get_config()
        
        # Model selection
        self.teacher_model = "qwen2.5:7b"  # Larger, more capable
        self.student_model = "deepseek-r1:1.5b"  # Smaller, faster
        
        # Initialize modules
        self.teacher = QueryRoutingTeacher()
        self.student = QueryRoutingStudent()
        
    def setup_teacher(self):
        """Configure DSPy with teacher model."""
        print(f"🧑‍🏫 Setting up teacher model: {self.teacher_model}")
        
        # Use proper DSPy BaseLM inheritance
        import litellm
        from dspy.clients.lm import BaseLM
        
        class LiteLLMWrapper(BaseLM):
            def __init__(self, model="ollama/qwen2.5:7b"):
                self.model = model
                self.kwargs = {"temperature": 0.1, "max_tokens": 500}
                self.history = []
                    
            def __call__(self, prompt=None, messages=None, **kwargs):
                try:
                    # Handle both prompt and messages format
                    if messages:
                        # Use messages directly
                        msgs = messages
                    elif prompt:
                        # Convert prompt to messages format
                        msgs = [{"role": "user", "content": prompt}]
                    else:
                        raise ValueError("Either prompt or messages must be provided")
                    
                    print(f"   📝 Teacher calling: {str(msgs)[:50]}...")
                    response = litellm.completion(
                        model=self.model,
                        messages=msgs,
                        api_base="http://localhost:11434",
                        temperature=0.1,
                        max_tokens=500
                    )
                    result = response.choices[0].message.content
                    print(f"   ✅ Teacher got: {result[:100]}...")
                    return result
                except Exception as e:
                    print(f"   ❌ LiteLLM call failed: {e}")
                    return '{"needs_video_search": "false", "needs_text_search": "false", "temporal_pattern": "none", "reasoning": "error"}'
        
        lm = LiteLLMWrapper(f"ollama/{self.teacher_model}")
        dspy.settings.configure(lm=lm)
        print("   ✅ Teacher configured with LiteLLM wrapper")
        
    def setup_student(self):
        """Configure DSPy with student model."""
        print(f"🎓 Setting up student model: {self.student_model}")
        
        import litellm
        from dspy.clients.lm import BaseLM
        
        class LiteLLMWrapper(BaseLM):
            def __init__(self, model="ollama/deepseek-r1:1.5b"):
                self.model = model
                self.kwargs = {"temperature": 0.1, "max_tokens": 300}
                self.history = []
                    
            def __call__(self, prompt=None, messages=None, **kwargs):
                try:
                    # Handle both prompt and messages format
                    if messages:
                        # Use messages directly
                        msgs = messages
                    elif prompt:
                        # Convert prompt to messages format
                        msgs = [{"role": "user", "content": prompt}]
                    else:
                        raise ValueError("Either prompt or messages must be provided")
                    
                    print(f"   📝 Student calling: {str(msgs)[:50]}...")
                    response = litellm.completion(
                        model=self.model,
                        messages=msgs,
                        api_base="http://localhost:11434",
                        temperature=0.1,
                        max_tokens=300
                    )
                    result = response.choices[0].message.content
                    print(f"   ✅ Student got: {result[:100]}...")
                    return result
                except Exception as e:
                    print(f"   ❌ LiteLLM call failed: {e}")
                    return '{"needs_video_search": "false", "needs_text_search": "false", "temporal_pattern": "none", "reasoning": "error"}'
        
        lm = LiteLLMWrapper(f"ollama/{self.student_model}")
        dspy.settings.configure(lm=lm)
        print("   ✅ Student configured with LiteLLM wrapper")
    
    async def generate_training_data(self) -> List[dspy.Example]:
        """Generate high-quality labeled data using teacher model."""
        
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
            "latest research"
        ]
        
        print(f"\n📝 Generating training data from {len(queries)} queries...")
        
        self.setup_teacher()
        training_examples = []
        
        for i, query in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] Query: '{query}'")
            
            try:
                # Get teacher's structured output
                teacher_output = self.teacher(query)
                
                # Parse boolean values properly
                needs_video = str(teacher_output.needs_video_search).lower() == "true"
                needs_text = str(teacher_output.needs_text_search).lower() == "true"
                temporal = teacher_output.temporal_pattern
                
                # Create training example
                example = dspy.Example(
                    query=query,
                    needs_video_search=needs_video,
                    needs_text_search=needs_text,
                    temporal_pattern=temporal,
                    reasoning=teacher_output.reasoning
                ).with_inputs("query")
                
                training_examples.append(example)
                
                print(f"   ✅ Video: {needs_video}, Text: {needs_text}, Temporal: {temporal}")
                print(f"   📝 Reasoning: {teacher_output.reasoning}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        return training_examples
    
    async def distill_knowledge(self, training_examples: List[dspy.Example]):
        """Distill teacher's knowledge into student model using MIPROv2."""
        
        print(f"\n🔄 Distilling knowledge to {self.student_model} with MIPROv2...")
        print(f"   Training examples: {len(training_examples)}")
        
        self.setup_student()
        
        # Use MIPROv2 - the best DSPy optimizer
        from dspy.teleprompt import MIPROv2
        
        def routing_accuracy(example, prediction, trace=None):
            """Measure agreement with teacher (distillation metric)."""
            score = 0.0
            
            # Parse prediction booleans
            pred_video = str(prediction.needs_video_search).lower() == "true"
            pred_text = str(prediction.needs_text_search).lower() == "true"
            
            # Check routing accuracy against teacher labels
            if pred_video == example.needs_video_search:
                score += 0.4
            if pred_text == example.needs_text_search:
                score += 0.4
            if prediction.temporal_pattern == example.temporal_pattern:
                score += 0.2
                
            return score
        
        # MIPROv2 optimizer - state of the art
        optimizer = MIPROv2(
            metric=routing_accuracy,
            prompt_model=None,  # Use configured student LM
            task_model=None,    # Use configured student LM
            num_candidates=15,   # More instruction candidates for better optimization
            init_temperature=1.4,
            verbose=True
        )
        
        print("   🧠 Running MIPROv2 optimization...")
        print("   📝 MIPROv2 will:")
        print("      - Generate multiple instruction candidates")
        print("      - Optimize prompts automatically")
        print("      - Bootstrap few-shot examples")
        print("      - Multi-stage optimization")
        
        start_time = time.time()
        
        # Split data for MIPROv2
        train_size = int(0.8 * len(training_examples))
        train_set = training_examples[:train_size]
        val_set = training_examples[train_size:]
        
        if len(val_set) == 0:
            val_set = train_set[-2:]  # Use last 2 examples for validation
        
        print(f"   📊 Train: {len(train_set)}, Validation: {len(val_set)}")
        
        optimized_student = optimizer.compile(
            self.student,
            trainset=train_set,
            num_trials=25,  # More trials for better optimization
            max_bootstrapped_demos=4,  # Optimal few-shot examples
            max_labeled_demos=8,       # Training examples to consider
            requires_permission_to_run=False  # Auto-run optimization
        )
        
        optimization_time = time.time() - start_time
        print(f"   ✅ MIPROv2 optimization complete in {optimization_time:.1f}s")
        
        # Test on validation set
        print("\n   📊 Validation Results:")
        val_correct = 0
        for example in val_set:
            try:
                pred = optimized_student(example.query)
                score = routing_accuracy(example, pred)
                if score >= 0.8:  # Consider >= 80% match as correct
                    val_correct += 1
            except:
                pass
        
        val_accuracy = val_correct / len(val_set) if val_set else 0
        print(f"   📈 Validation accuracy: {val_accuracy:.1%}")
        
        return optimized_student
    
    def save_distilled_model(self, student: QueryRoutingStudent, accuracy: float):
        """Save the distilled model configuration."""
        
        config = {
            "model": self.student_model,
            "accuracy": accuracy,
            "teacher_model": self.teacher_model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "optimizer": "MIPROv2",
            "dspy_config": {
                "optimizer": "MIPROv2",
                "num_candidates": 15,
                "max_bootstrapped_demos": 4,
                "max_labeled_demos": 8,
                "num_trials": 25,
                "signature": "StructuredRoutingSignature",
                "features": [
                    "Multi-stage instruction optimization",
                    "Automatic prompt generation",
                    "Few-shot example bootstrapping",
                    "Student-teacher distillation"
                ]
            }
        }
        
        with open("miprov2_distilled_router.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Saved MIPROv2 distilled model to miprov2_distilled_router.json")

async def main():
    """Run the complete distillation process."""
    
    print("🚀 DSPy Query Routing Distillation with Ollama")
    print("=" * 60)
    
    distiller = OllamaDistillation()
    
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
    
    distiller.setup_student()
    correct = 0
    
    for query, exp_video, exp_text, exp_temporal in test_queries:
        try:
            result = optimized_student(query)
            
            pred_video = str(result.needs_video_search).lower() == "true"
            pred_text = str(result.needs_text_search).lower() == "true"
            
            is_correct = (
                pred_video == exp_video and 
                pred_text == exp_text and
                result.temporal_pattern == exp_temporal
            )
            
            if is_correct:
                correct += 1
                
            print(f"\n🔍 Query: '{query}'")
            print(f"   Expected: video={exp_video}, text={exp_text}, temporal={exp_temporal}")
            print(f"   Got: video={pred_video}, text={pred_text}, temporal={result.temporal_pattern}")
            print(f"   {'✅ Correct' if is_correct else '❌ Wrong'}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    accuracy = correct / len(test_queries)
    print(f"\n📈 Student Accuracy: {accuracy:.1%}")
    
    # Save the model
    distiller.save_distilled_model(optimized_student, accuracy)
    
    print("\n✅ Distillation complete!")
    print(f"   Student model: {distiller.student_model}")
    print(f"   Can now handle routing with structured outputs!")

if __name__ == "__main__":
    asyncio.run(main())