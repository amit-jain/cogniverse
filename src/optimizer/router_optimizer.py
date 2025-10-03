"""
Agentic RAG Router Optimizer using DSPy MIPROv2

This optimizer implements the design specified in NEW_PROPOSAL.md:
- Uses google/gemma-3-1b-it as the student model
- Implements unified decision schema with search_modality and generation_type
- Uses DSPy MIPROv2 for optimization
- Outputs portable prompt artifacts for agent integration
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path

import dspy
from dspy.teleprompt import MIPROv2
import pandas as pd
from dotenv import load_dotenv

# Import schemas from the new location
from .schemas import RoutingDecision, AgenticRouter

# Load environment variables
load_dotenv()


# ==================== DSPy Module ====================

class RouterModule(dspy.Module):
    """DSPy module for the routing agent."""
    
    def __init__(self):
        super().__init__()
        self.route = dspy.Predict(AgenticRouter)
    
    def forward(self, conversation_history: str, user_query: str) -> RoutingDecision:
        """Execute the routing decision."""
        print(f"\n🔍 RouterModule.forward called with query: {user_query[:100]}...")
        
        response = self.route(
            conversation_history=conversation_history,
            user_query=user_query
        )
        
        # Log the raw response for debugging
        print(f"📝 Raw LLM response: {response}")
        if hasattr(response, '_asdict'):
            print(f"📊 Response dict: {response._asdict()}")
        
        return response.routing_decision


# ==================== Dataset Generation ====================

def generate_training_examples() -> List[dspy.Example]:
    """
    Generate diverse training examples for the router.
    These examples cover different combinations of search modalities and generation types.
    """
    examples = []
    
    # Video + Raw Results
    examples.extend([
        {
            "conversation_history": "",
            "user_query": "Show me how to bake a cake",
            "routing_decision": {"search_modality": "video", "generation_type": "raw_results"}
        },
        {
            "conversation_history": "",
            "user_query": "Find me tutorials on Python programming",
            "routing_decision": {"search_modality": "video", "generation_type": "raw_results"}
        },
        {
            "conversation_history": "User asked about cooking earlier",
            "user_query": "Show me the pasta making video",
            "routing_decision": {"search_modality": "video", "generation_type": "raw_results"}
        }
    ])
    
    # Text + Detailed Report
    examples.extend([
        {
            "conversation_history": "",
            "user_query": "Construct a detailed report on the economic impact of AI",
            "routing_decision": {"search_modality": "text", "generation_type": "detailed_report"}
        },
        {
            "conversation_history": "",
            "user_query": "Create a comprehensive analysis of climate change policies",
            "routing_decision": {"search_modality": "text", "generation_type": "detailed_report"}
        },
        {
            "conversation_history": "User is researching renewable energy",
            "user_query": "Build a detailed report on solar panel efficiency",
            "routing_decision": {"search_modality": "text", "generation_type": "detailed_report"}
        }
    ])
    
    # Text + Summary
    examples.extend([
        {
            "conversation_history": "",
            "user_query": "What is the main point of the new climate study?",
            "routing_decision": {"search_modality": "text", "generation_type": "summary"}
        },
        {
            "conversation_history": "",
            "user_query": "Summarize the key findings about vaccine effectiveness",
            "routing_decision": {"search_modality": "text", "generation_type": "summary"}
        },
        {
            "conversation_history": "User read about quantum computing",
            "user_query": "Give me a quick summary of quantum supremacy",
            "routing_decision": {"search_modality": "text", "generation_type": "summary"}
        }
    ])
    
    # Video + Summary (less common but valid)
    examples.extend([
        {
            "conversation_history": "",
            "user_query": "Summarize the main points from the TED talk on leadership",
            "routing_decision": {"search_modality": "video", "generation_type": "summary"}
        },
        {
            "conversation_history": "User watched a documentary",
            "user_query": "What were the key takeaways from that video?",
            "routing_decision": {"search_modality": "video", "generation_type": "summary"}
        }
    ])
    
    # Text + Raw Results
    examples.extend([
        {
            "conversation_history": "",
            "user_query": "Find research papers on machine learning",
            "routing_decision": {"search_modality": "text", "generation_type": "raw_results"}
        },
        {
            "conversation_history": "",
            "user_query": "Show me articles about space exploration",
            "routing_decision": {"search_modality": "text", "generation_type": "raw_results"}
        }
    ])
    
    # Convert to DSPy Examples with proper RoutingDecision objects
    dspy_examples = []
    for ex in examples:
        dspy_examples.append(dspy.Example(
            conversation_history=ex["conversation_history"],
            user_query=ex["user_query"],
            routing_decision=RoutingDecision(**ex["routing_decision"])
        ).with_inputs("conversation_history", "user_query"))
    
    return dspy_examples


def generate_teacher_examples(teacher_lm: dspy.LM, num_examples: int = 50) -> List[dspy.Example]:
    """
    Use a teacher model to generate high-quality training examples.
    
    Args:
        teacher_lm: The teacher language model (e.g., Claude, GPT-4)
        num_examples: Number of examples to generate
    
    Returns:
        List of DSPy examples
    """
    # Set the teacher as the default LM temporarily
    with dspy.context(lm=teacher_lm):
        router = RouterModule()
        
        # Generate diverse queries
        query_templates = [
            # Video queries
            "Show me how to {action}",
            "Find tutorials on {topic}",
            "I want to watch videos about {subject}",
            "Play the {video_name} video",
            
            # Text + Report queries
            "Create a detailed report on {topic}",
            "Construct a comprehensive analysis of {subject}",
            "Build an in-depth report about {area}",
            "Generate a thorough report on {domain}",
            
            # Text + Summary queries
            "What is the main point of {document}?",
            "Summarize the key findings about {topic}",
            "Give me a quick overview of {subject}",
            "What are the highlights of {study}?",
            
            # Raw results queries
            "Find {content_type} about {topic}",
            "Show me {content_type} on {subject}",
            "Search for {content_type} related to {area}"
        ]
        
        # Topic variations
        topics = [
            "artificial intelligence", "climate change", "quantum computing",
            "renewable energy", "space exploration", "biotechnology",
            "cryptocurrency", "mental health", "sustainable agriculture",
            "robotics", "gene therapy", "blockchain", "machine learning",
            "virtual reality", "cybersecurity", "nanotechnology"
        ]
        
        actions = [
            "cook pasta", "fix a bicycle", "learn guitar", "code in Python",
            "meditate", "grow vegetables", "build a website", "paint watercolors"
        ]
        
        examples = []
        
        # Generate examples using the teacher
        for i in range(num_examples):
            # Randomly select template and fill it
            import random
            template = random.choice(query_templates)
            
            if "{action}" in template:
                query = template.format(action=random.choice(actions))
            elif "{topic}" in template or "{subject}" in template or "{area}" in template or "{domain}" in template:
                query = template.format(
                    topic=random.choice(topics),
                    subject=random.choice(topics),
                    area=random.choice(topics),
                    domain=random.choice(topics)
                )
            elif "{content_type}" in template:
                content_type = random.choice(["papers", "articles", "documents", "videos"])
                query = template.format(
                    content_type=content_type,
                    topic=random.choice(topics),
                    subject=random.choice(topics),
                    area=random.choice(topics)
                )
            else:
                query = template.format(
                    video_name=f"{random.choice(['tutorial', 'lecture', 'presentation'])} {random.choice(topics)}",
                    document=f"the {random.choice(['paper', 'article', 'study'])} on {random.choice(topics)}",
                    study=f"{random.choice(topics)} research"
                )
            
            # Generate conversation history sometimes
            conversation_history = ""
            if random.random() > 0.7:  # 30% chance of having history
                conversation_history = f"User previously asked about {random.choice(topics)}"
            
            try:
                # Get routing decision from teacher
                result = router(conversation_history=conversation_history, user_query=query)
                
                examples.append(dspy.Example(
                    conversation_history=conversation_history,
                    user_query=query,
                    routing_decision=result
                ).with_inputs("conversation_history", "user_query"))
                
            except Exception as e:
                print(f"Error generating example {i}: {e}")
                continue
        
        return examples


# ==================== Evaluation Metrics ====================

def evaluate_routing_accuracy(module: RouterModule, test_set: List[dspy.Example]) -> Dict[str, float]:
    """
    Evaluate the routing accuracy of the module.
    
    Returns:
        Dictionary with accuracy metrics
    """
    correct_modality = 0
    correct_generation = 0
    correct_both = 0
    total = len(test_set)
    
    for example in test_set:
        try:
            prediction = module(
                conversation_history=example.conversation_history,
                user_query=example.user_query
            )
            
            # Check modality accuracy
            if prediction.search_modality == example.routing_decision.search_modality:
                correct_modality += 1
            
            # Check generation type accuracy
            if prediction.generation_type == example.routing_decision.generation_type:
                correct_generation += 1
            
            # Check both correct
            if (prediction.search_modality == example.routing_decision.search_modality and
                prediction.generation_type == example.routing_decision.generation_type):
                correct_both += 1
                
        except Exception as e:
            print(f"Error evaluating example: {e}")
            continue
    
    return {
        "modality_accuracy": correct_modality / total if total > 0 else 0,
        "generation_accuracy": correct_generation / total if total > 0 else 0,
        "overall_accuracy": correct_both / total if total > 0 else 0,
        "total_examples": total
    }


# ==================== Main Optimization Function ====================

def optimize_router(
    student_model: str = "google/gemma-3-1b-it",
    teacher_model: Optional[str] = None,
    use_manual_examples: bool = True,
    num_teacher_examples: int = 50,
    output_dir: str = "optimization_results"
) -> Dict:
    """
    Run the MIPROv2 optimization for the router.
    
    Args:
        student_model: Model ID for the student (default: google/gemma-3-1b-it)
        teacher_model: Model ID for the teacher (optional, for generating examples)
        use_manual_examples: Whether to use manually crafted examples
        num_teacher_examples: Number of examples to generate with teacher
        output_dir: Directory to save results
    
    Returns:
        Dictionary with optimization results
    """
    # Use OutputManager for organized output
    from src.utils.output_manager import get_output_manager
    output_manager = get_output_manager()
    output_path = output_manager.get_optimization_dir()
    
    # Initialize models
    print(f"Initializing student model: {student_model}")
    
    # Configure student LM based on the model type
    if "gemma" in student_model.lower():
        # For Gemma models, we'll use the appropriate provider
        # This could be HuggingFace, Together, or another provider
        student_lm = dspy.LM(
            model=student_model,
            temperature=0.1,  # Low temperature for consistent JSON output
            max_tokens=100,   # Router output is small
        )
    else:
        # Default configuration for other models
        student_lm = dspy.LM(
            model=student_model,
            temperature=0.1,
            max_tokens=100,
        )
    
    dspy.configure(lm=student_lm)
    
    # Generate or load training data
    print("Generating training data...")
    if use_manual_examples:
        train_examples = generate_training_examples()
        print(f"Generated {len(train_examples)} manual examples")
    else:
        train_examples = []
    
    if teacher_model and num_teacher_examples > 0:
        print(f"Generating {num_teacher_examples} examples with teacher model: {teacher_model}")
        
        # Configure teacher LM based on model type
        if "claude" in teacher_model.lower():
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            teacher_lm = dspy.LM(
                model=teacher_model,
                api_key=api_key,
                temperature=0.7
            )
        elif "gpt" in teacher_model.lower():
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            teacher_lm = dspy.LM(
                model=teacher_model,
                api_key=api_key,
                temperature=0.7
            )
        else:
            # Default configuration
            teacher_lm = dspy.LM(model=teacher_model, temperature=0.7)
        
        teacher_examples = generate_teacher_examples(teacher_lm, num_teacher_examples)
        train_examples.extend(teacher_examples)
        print(f"Total examples: {len(train_examples)}")
    
    # Split into train and validation
    split_idx = int(0.8 * len(train_examples))
    train_set = train_examples[:split_idx]
    val_set = train_examples[split_idx:]
    
    print(f"Train set: {len(train_set)}, Validation set: {len(val_set)}")
    
    # Define the metric for optimization
    def routing_metric(example: dspy.Example, prediction: RoutingDecision) -> float:
        """Metric for MIPROv2 optimization."""
        score = 0.0
        
        # Check if prediction is valid
        if not isinstance(prediction, RoutingDecision):
            return 0.0
        
        # Check modality (50% weight)
        if prediction.search_modality == example.routing_decision.search_modality:
            score += 0.5
        
        # Check generation type (50% weight)
        if prediction.generation_type == example.routing_decision.generation_type:
            score += 0.5
        
        return score
    
    # Initialize router module
    router = RouterModule()
    
    # Run baseline evaluation
    print("\nEvaluating baseline performance...")
    baseline_metrics = evaluate_routing_accuracy(router, val_set)
    print(f"Baseline accuracy: {baseline_metrics}")
    
    # Run MIPROv2 optimization
    print("\nRunning MIPROv2 optimization...")
    start_time = time.time()
    
    optimizer = MIPROv2(
        metric=routing_metric,
        num_candidates=10,  # Number of instruction candidates
        init_temperature=0.7,
        verbose=True
    )
    
    # Compile the module
    optimized_router = optimizer.compile(
        router,
        trainset=train_set,
        valset=val_set,
        num_trials=20,  # Number of optimization trials
        minibatch_size=4,
        minibatch_full_eval_steps=10,
        minibatch=True,
        requires_permission_to_run=False
    )
    
    optimization_time = time.time() - start_time
    
    # Evaluate optimized performance
    print("\nEvaluating optimized performance...")
    optimized_metrics = evaluate_routing_accuracy(optimized_router, val_set)
    print(f"Optimized accuracy: {optimized_metrics}")
    
    # Extract optimized prompt and demonstrations
    print("\nExtracting optimization artifacts...")
    
    # Get the optimized state
    state = optimized_router.dump_state()
    
    # Extract demonstrations and instructions
    artifacts = {
        "instructions": state.get("route", {}).get("instructions", ""),
        "demonstrations": [],
        "model_config": {
            "student_model": student_model,
            "temperature": 0.1,
            "max_tokens": 100
        },
        "metrics": {
            "baseline": baseline_metrics,
            "optimized": optimized_metrics,
            "improvement": {
                "modality": optimized_metrics["modality_accuracy"] - baseline_metrics["modality_accuracy"],
                "generation": optimized_metrics["generation_accuracy"] - baseline_metrics["generation_accuracy"],
                "overall": optimized_metrics["overall_accuracy"] - baseline_metrics["overall_accuracy"]
            }
        },
        "optimization_time": optimization_time,
        "timestamp": datetime.now().isoformat()
    }
    
    # Extract demonstrations
    if "demos" in state.get("route", {}):
        for demo in state["route"]["demos"]:
            artifacts["demonstrations"].append({
                "conversation_history": demo.get("conversation_history", ""),
                "user_query": demo.get("user_query", ""),
                "routing_decision": {
                    "search_modality": demo.get("routing_decision", {}).get("search_modality", ""),
                    "generation_type": demo.get("routing_decision", {}).get("generation_type", "")
                }
            })
    
    # Save artifacts
    output_file = output_path / f"router_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(artifacts, f, indent=2)
    
    print(f"\nOptimization complete! Results saved to: {output_file}")
    
    # Save a separate file for easy integration
    integration_file = output_path / "router_prompt_artifact.json"
    integration_data = {
        "system_prompt": artifacts["instructions"],
        "few_shot_examples": artifacts["demonstrations"],
        "model_config": artifacts["model_config"]
    }
    with open(integration_file, "w") as f:
        json.dump(integration_data, f, indent=2)
    
    print(f"Integration artifact saved to: {integration_file}")
    
    return artifacts


# ==================== Integration Helpers ====================

class OptimizedRouter:
    """
    Production-ready router using the optimized artifacts.
    This can be integrated with Letta and ADK/A2A.
    """
    
    def __init__(self, artifact_path: str):
        """Load the optimized artifacts."""
        with open(artifact_path, "r") as f:
            self.artifacts = json.load(f)
        
        # Initialize the LM with optimized config
        config = self.artifacts["model_config"]
        self.lm = dspy.LM(
            model=config["student_model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )
        
        # Build the prompt template
        self.system_prompt = self.artifacts["system_prompt"]
        self.examples = self.artifacts["few_shot_examples"]
    
    def route(self, user_query: str, conversation_history: str = "") -> RoutingDecision:
        """
        Make a routing decision using the optimized prompt.
        
        Args:
            user_query: The user's query
            conversation_history: Recent conversation context
        
        Returns:
            RoutingDecision object
        """
        # Build the full prompt with few-shot examples
        prompt = self.system_prompt + "\n\n"
        
        # Add few-shot examples
        if self.examples:
            prompt += "Examples:\n"
            for ex in self.examples[:3]:  # Use top 3 examples
                prompt += f"\nConversation History: {ex['conversation_history']}\n"
                prompt += f"User Query: {ex['user_query']}\n"
                prompt += f"Output: {json.dumps(ex['routing_decision'])}\n"
        
        # Add the current query
        prompt += f"\nConversation History: {conversation_history}\n"
        prompt += f"User Query: {user_query}\n"
        prompt += "Output: "
        
        # Get response from LM
        response = self.lm(prompt)
        
        # Parse JSON response
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                decision_dict = json.loads(json_match.group())
                return RoutingDecision(**decision_dict)
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"Error parsing response: {e}")
            # Fallback to sensible defaults
            return RoutingDecision(
                search_modality="text",
                generation_type="raw_results"
            )


# ==================== CLI Interface ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize the Agentic RAG Router")
    parser.add_argument("--student-model", default="google/gemma-3-1b-it",
                       help="Student model to optimize")
    parser.add_argument("--teacher-model", default=None,
                       help="Teacher model for generating examples (optional)")
    parser.add_argument("--num-examples", type=int, default=50,
                       help="Number of examples to generate with teacher")
    parser.add_argument("--output-dir", default="optimization_results",
                       help="Directory to save results")
    parser.add_argument("--no-manual-examples", action="store_true",
                       help="Don't use manually crafted examples")
    
    args = parser.parse_args()
    
    # Run optimization
    results = optimize_router(
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        use_manual_examples=not args.no_manual_examples,
        num_teacher_examples=args.num_examples,
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Student Model: {args.student_model}")
    print(f"Baseline Overall Accuracy: {results['metrics']['baseline']['overall_accuracy']:.2%}")
    print(f"Optimized Overall Accuracy: {results['metrics']['optimized']['overall_accuracy']:.2%}")
    print(f"Improvement: {results['metrics']['improvement']['overall']:.2%}")
    print(f"Optimization Time: {results['optimization_time']:.2f} seconds")