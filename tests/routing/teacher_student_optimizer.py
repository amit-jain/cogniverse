#!/usr/bin/env python3
# tests/teacher_student_optimizer.py
"""
Teacher-Student DSPy optimization using a larger teacher model
to improve the smaller student model performance.
"""

import asyncio
import sys
import os
import dspy
from typing import List, Dict, Any, Tuple
import json
import re

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from tools.query_analyzer import QueryAnalyzer
from tools.config import get_config

def create_query_example(query, needs_video, needs_text, temporal_pattern=None):
    """Create a properly formatted DSPy example."""
    example = dspy.Example(
        query=query,
        needs_video_search=needs_video,
        needs_text_search=needs_text,
        temporal_pattern=temporal_pattern if temporal_pattern else "null"
    ).with_inputs("query")
    
    # Add compatibility attributes for our metric
    example.needs_video = needs_video
    example.needs_text = needs_text
    example.temporal_pattern = temporal_pattern
    
    return example

def setup_teacher_student_llms(teacher_model="qwen2.5:7b", student_model="qwen3:1.7b", base_url="http://localhost:11434"):
    """Set up teacher and student LLMs for DSPy."""
    
    # Check if models are available
    import requests
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            raise Exception(f"Ollama server returned status {response.status_code}")
        
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        
        print(f"📋 Available models: {model_names[:5]}")
        
        # Use the specified teacher model directly
        if teacher_model in model_names:
            teacher_found = teacher_model
            print(f"✅ Using specified teacher model: {teacher_model}")
        else:
            print(f"⚠️ Teacher model {teacher_model} not found. Using {student_model} for both.")
            teacher_found = student_model
        
        if student_model not in model_names:
            print(f"⚠️ Student model {student_model} not found. Available: {model_names[:3]}")
            # Use the smallest available model as student
            student_model = model_names[0] if model_names else student_model
            
    except Exception as e:
        print(f"⚠️ Cannot check models: {e}")
        teacher_found = teacher_model
    
    # Configure teacher model (larger, more capable)
    teacher_lm = dspy.LM(
        model=f"ollama/{teacher_found}",
        api_base=base_url,
        api_key="dummy",
        model_type="chat",
        max_tokens=400,
        temperature=0.1
    )
    
    # Configure student model (smaller, to be optimized)
    student_lm = dspy.LM(
        model=f"ollama/{student_model}",
        api_base=base_url,
        api_key="dummy", 
        model_type="chat",
        max_tokens=300,
        temperature=0.1
    )
    
    print(f"🎓 Teacher Model: {teacher_found}")
    print(f"👨‍🎓 Student Model: {student_model}")
    
    return teacher_lm, student_lm

class QueryRouterSignature(dspy.Signature):
    """Query routing signature for both teacher and student."""
    query = dspy.InputField(desc="User query to analyze for routing and temporal patterns")
    needs_video_search = dspy.OutputField(desc="Answer 'true' or 'false': does the query need video/visual content search?")
    needs_text_search = dspy.OutputField(desc="Answer 'true' or 'false': does the query need text/document search?")
    temporal_pattern = dspy.OutputField(desc="Extract temporal pattern from query (like 'yesterday', 'last_week') or answer 'null' if none found")
    reasoning = dspy.OutputField(desc="Brief explanation of your routing and temporal decisions")

class QueryRouter(dspy.Module):
    """Base query router for both teacher and student."""
    
    def __init__(self):
        super().__init__()
        self.router = dspy.Predict(QueryRouterSignature)
    
    def forward(self, query: str):
        """Process a query and return routing decision."""
        try:
            result = self.router(query=query)
            
            # Parse boolean outputs robustly
            needs_video = str(result.needs_video_search).lower() in ['true', '1', 'yes']
            needs_text = str(result.needs_text_search).lower() in ['true', '1', 'yes']
            
            # Clean temporal pattern
            temporal = result.temporal_pattern
            if temporal and temporal.lower() in ['null', 'none', 'no', '']:
                temporal = None
            
            return dspy.Prediction(
                needs_video_search=needs_video,
                needs_text_search=needs_text,
                temporal_pattern=temporal,
                reasoning=getattr(result, 'reasoning', 'No reasoning provided')
            )
            
        except Exception as e:
            print(f"⚠️ Router prediction failed: {e}")
            # Return default prediction
            return dspy.Prediction(
                needs_video_search=False,
                needs_text_search=True,
                temporal_pattern=None,
                reasoning="Error in prediction"
            )

class QueryAnalysisMetric:
    """Metric for evaluating query analysis performance."""
    
    def __call__(self, example, prediction, trace=None) -> float:
        """Calculate accuracy score for a prediction."""
        score = 0.0
        total_weight = 3.0
        
        # Get expected values from example
        expected_video = getattr(example, 'needs_video_search', getattr(example, 'needs_video', False))
        expected_text = getattr(example, 'needs_text_search', getattr(example, 'needs_text', False))
        expected_temporal = getattr(example, 'temporal_pattern', None)
        
        # Get actual values from prediction
        actual_video = getattr(prediction, 'needs_video_search', False)
        actual_text = getattr(prediction, 'needs_text_search', False)
        actual_temporal = getattr(prediction, 'temporal_pattern', None)
        
        # Handle null temporal pattern
        if expected_temporal == "null":
            expected_temporal = None
        if actual_temporal == "null":
            actual_temporal = None
        
        # Routing accuracy (2 components)
        if actual_video == expected_video:
            score += 1.0
        if actual_text == expected_text:
            score += 1.0
        
        # Temporal accuracy (1 component)
        if actual_temporal == expected_temporal:
            score += 1.0
        
        return score / total_weight

def load_training_examples():
    """Load training examples from test queries."""
    queries = []
    test_file = os.path.join(os.path.dirname(__file__), "test_queries.txt")
    
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                parts = line.split(',', 2)
                if len(parts) != 3:
                    continue
                
                query = parts[0].strip()
                expected_routing = parts[1].strip()
                expected_temporal = parts[2].strip()
                
                # Convert routing to boolean format
                if expected_routing == "video":
                    needs_video, needs_text = True, False
                elif expected_routing == "text":
                    needs_video, needs_text = False, True
                elif expected_routing == "both":
                    needs_video, needs_text = True, True
                else:
                    continue
                
                # Handle temporal pattern
                if expected_temporal.lower() in ["none", "null", ""]:
                    expected_temporal = None
                
                queries.append(create_query_example(
                    query=query,
                    needs_video=needs_video,
                    needs_text=needs_text,
                    temporal_pattern=expected_temporal
                ))
                
            except Exception as e:
                continue
    
    return queries

def create_high_quality_demonstrations():
    """Create high-quality demonstration examples for DSPy."""
    return [
        # Clear video-only examples
        create_query_example("Show me videos of cats playing", True, False, None),
        create_query_example("Find footage of the meeting", True, False, None),
        create_query_example("Display video recordings", True, False, None),
        
        # Clear text-only examples  
        create_query_example("Find documents about AI research", False, True, None),
        create_query_example("Search for reports on quarterly performance", False, True, None),
        create_query_example("Look up articles about machine learning", False, True, None),
        
        # Multi-modal examples
        create_query_example("Find videos and documents about training", True, True, None),
        create_query_example("Get all content about the project", True, True, None),
        
        # Temporal examples
        create_query_example("Show me videos from yesterday", True, False, "yesterday"),
        create_query_example("Find reports from last week", False, True, "last_week"),
        create_query_example("Get content from this month", True, True, "this_month"),
    ]

async def teacher_student_optimization():
    """Run teacher-student optimization with DSPy."""
    print("🎓 Teacher-Student MIPROv2 Optimization")
    print("=" * 50)
    print("Using qwen3:8b teacher → qwen3:1.7b student with MIPROv2")
    
    try:
        # Set up teacher and student models - qwen3:8b teacher, qwen3:1.7b student
        print("🔧 Setting up qwen3:8b teacher and qwen3:1.7b student...")
        base_url = "http://localhost:11434"
        
        teacher_lm = dspy.LM(
            model="ollama/qwen3:8b",  # Larger teacher model
            api_base=base_url,
            api_key="dummy",
            model_type="chat",
            max_tokens=300,
            temperature=0.0  # Teacher should be deterministic
        )
        
        student_lm = dspy.LM(
            model="ollama/qwen3:1.7b",  # Smaller student model
            api_base=base_url,
            api_key="dummy",
            model_type="chat",
            max_tokens=200,
            temperature=0.1  # Student can have slight variation
        )
        
        print(f"🎓 Teacher: qwen3:8b (larger, more capable)")
        print(f"👨‍🎓 Student: qwen3:1.7b (smaller, to be optimized)")
        
        # Load training data
        examples = load_training_examples()
        train_examples = examples[:20]  # Larger training set for teacher-student
        test_examples = examples[50:60]
        
        # Create high-quality demonstrations
        demos = create_high_quality_demonstrations()
        
        print(f"📚 Training setup:")
        print(f"   - {len(demos)} demonstrations")
        print(f"   - {len(train_examples)} training examples")  
        print(f"   - {len(test_examples)} test examples")
        
        # Create teacher router with larger model
        print(f"\n🎓 Setting up teacher model...")
        dspy.settings.configure(lm=teacher_lm)
        teacher_router = QueryRouter()
        metric = QueryAnalysisMetric()
        
        # Evaluate teacher baseline
        print(f"\n📊 Evaluating teacher performance...")
        teacher_score = 0
        teacher_details = {"correct": 0, "total": 0}
        
        for i, example in enumerate(test_examples[:5]):  # Test subset for teacher
            try:
                prediction = teacher_router(query=example.query)
                score = metric(example, prediction)
                teacher_score += score
                teacher_details["total"] += 1
                if score == 1.0:
                    teacher_details["correct"] += 1
                print(f"   Teacher test {i+1}: {score:.2f}")
            except Exception as e:
                print(f"   Teacher test {i+1}: Error - {e}")
        
        if teacher_details["total"] > 0:
            teacher_score /= teacher_details["total"]
            print(f"\n📊 Teacher Performance: {teacher_score:.1%} ({teacher_details['correct']}/{teacher_details['total']} perfect)")
        else:
            print("❌ Teacher evaluation failed")
            return None, 0.0
        
        # Generate teacher predictions for training data
        print(f"\n🎓 Generating teacher predictions from qwen3:8b...")
        teacher_labeled_data = []
        
        dspy.settings.configure(lm=teacher_lm)
        for i, example in enumerate(train_examples):
            try:
                # Get teacher prediction from larger model
                teacher_prediction = teacher_router(query=example.query)
                
                # Create labeled example with teacher output
                labeled_example = dspy.Example(
                    query=example.query,
                    needs_video_search=teacher_prediction.needs_video_search,
                    needs_text_search=teacher_prediction.needs_text_search,
                    temporal_pattern=teacher_prediction.temporal_pattern,
                    reasoning=getattr(teacher_prediction, 'reasoning', 'Teacher prediction')
                ).with_inputs("query")
                
                teacher_labeled_data.append(labeled_example)
                if i % 5 == 0:
                    print(f"   Generated {i+1}/{len(train_examples)} teacher labels...")
                    
            except Exception as e:
                print(f"   Error generating teacher label {i+1}: {e}")
                continue
        
        print(f"✅ Generated {len(teacher_labeled_data)} teacher-labeled examples")
        
        # Now optimize student model with teacher data
        print(f"\n👨‍🎓 Optimizing student model...")
        dspy.settings.configure(lm=student_lm)
        student_router = QueryRouter()
        
        # Evaluate student baseline
        print(f"\n📊 Evaluating student baseline...")
        student_baseline_score = 0
        student_baseline_details = {"correct": 0, "total": 0}
        
        for i, example in enumerate(test_examples):
            try:
                prediction = student_router(query=example.query)
                score = metric(example, prediction)
                student_baseline_score += score
                student_baseline_details["total"] += 1
                if score == 1.0:
                    student_baseline_details["correct"] += 1
                print(f"   Student baseline {i+1}: {score:.2f}")
            except Exception as e:
                print(f"   Student baseline {i+1}: Error - {e}")
        
        if student_baseline_details["total"] > 0:
            student_baseline_score /= student_baseline_details["total"]
            print(f"\n📊 Student Baseline: {student_baseline_score:.1%}")
        else:
            print("❌ Student baseline evaluation failed")
            return None, 0.0
        
        # Run MIPROv2 optimization on student with teacher-labeled data
        print(f"\n🔧 Running MIPROv2 optimization on student...")
        
        from dspy.teleprompt import MIPROv2
        
        # Combine demos with teacher-labeled data
        full_trainset = demos + teacher_labeled_data
        
        teleprompter = MIPROv2(
            metric=metric,
            prompt_model=teacher_lm,  # Use teacher model for prompt optimization
            task_model=student_lm,    # Use student model for task execution
            auto="light",             # Use auto mode instead of manual parameters
            init_temperature=0.5      # Temperature for prompt generation
        )
        
        print(f"   📚 Training student with MIPROv2 using {len(full_trainset)} examples")
        print(f"   🎓 Using qwen3:8b for prompt optimization")
        print(f"   👨‍🎓 Training qwen3:1.7b for task execution")
        optimized_student_router = teleprompter.compile(student_router, trainset=full_trainset)
        
        # Evaluate optimized student
        print(f"\n📊 Evaluating optimized student...")
        optimized_score = 0
        optimized_details = {"correct": 0, "total": 0}
        
        for i, example in enumerate(test_examples):
            try:
                prediction = optimized_student_router(query=example.query)
                score = metric(example, prediction)
                optimized_score += score
                optimized_details["total"] += 1
                if score == 1.0:
                    optimized_details["correct"] += 1
                print(f"   Optimized student {i+1}: {score:.2f}")
            except Exception as e:
                print(f"   Optimized student {i+1}: Error - {e}")
        
        if optimized_details["total"] > 0:
            optimized_score /= optimized_details["total"]
            
            print(f"\n🎯 MIPROv2 TEACHER-STUDENT RESULTS:")
            print(f"   Teacher (qwen3:8b):     {teacher_score:.1%}")
            print(f"   Student Baseline:       {student_baseline_score:.1%}")
            print(f"   Student MIPROv2 Opt:    {optimized_score:.1%}")
            
            if optimized_score > student_baseline_score:
                improvement = ((optimized_score - student_baseline_score) / student_baseline_score * 100)
                print(f"   📈 MIPROv2 Improvement: +{improvement:.1f}%")
                
                # Compare to teacher
                if teacher_score > 0:
                    teacher_gap = ((teacher_score - optimized_score) / teacher_score * 100)
                    print(f"   📊 Gap to Teacher:      -{teacher_gap:.1f}%")
                
                print(f"   ✅ MIPROv2 optimization successful!")
                return optimized_student_router, optimized_score
            else:
                decline = ((student_baseline_score - optimized_score) / student_baseline_score * 100)
                print(f"   📉 Performance declined: -{decline:.1f}%")
                print(f"   🔄 Keeping baseline (MIPROv2 didn't help)")
                return student_router, student_baseline_score
        else:
            return student_router, student_baseline_score
                
    except Exception as e:
        print(f"❌ Teacher-student optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

async def main():
    """Run teacher-student optimization."""
    print("🚀 Teacher-Student MIPROv2 Optimization")
    print("=" * 60)
    print("Using qwen3:8b teacher model to optimize qwen3:1.7b student with MIPROv2")
    print()
    
    # Run teacher-student optimization
    optimized_router, final_score = await teacher_student_optimization()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEACHER-STUDENT MIPROv2 OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    if final_score and final_score > 0:
        print(f"🎯 Final Student Model Accuracy: {final_score:.1%}")
        
        if final_score >= 0.70:
            print("🏆 Excellent performance! MIPROv2 teacher-student worked well!")
        elif final_score >= 0.60:
            print("✅ Good performance - MIPROv2 optimization successful")
        elif final_score >= 0.50:
            print("⚠️ Moderate performance - some MIPROv2 improvement")
        else:
            print("❌ Limited improvement - may need different optimization approach")
            
        print(f"\n💡 MIPROv2 Teacher-Student Benefits:")
        print(f"   • Uses qwen3:8b for intelligent prompt optimization")
        print(f"   • Optimizes qwen3:1.7b for fast inference")
        print(f"   • Generates better prompts than BootstrapFewShot")
        print(f"   • Leverages teacher model knowledge in prompt design")
    else:
        print("❌ Teacher-student optimization failed")

if __name__ == "__main__":
    asyncio.run(main()) 