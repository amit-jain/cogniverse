#!/usr/bin/env python3
"""
Agentic Router Optimization Orchestrator

This script reads the configuration and:
1. Uses provider abstractions for model hosting and artifact storage
2. Calls the existing DSPy optimizer classes using provider API calls
3. Handles all model providers (modal, local, anthropic, openai) through abstractions
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Import all providers to register them
from .providers.base_provider import DSPyLMProvider, ProviderFactory
from .router_optimizer import RouterModule, evaluate_routing_accuracy

# Import from the new structure
from .schemas import RoutingDecision

# Load environment variables
load_dotenv()


# ==================== Provider-Based Client ====================

class ModelClient:
    """
    Client that can call any model through provider abstractions.
    Supports: modal, local (ollama), anthropic, openai
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_providers = {}
        self.artifact_providers = {}
        
        # Initialize providers based on configuration
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize providers based on configuration."""
        
        # Initialize model providers for teacher and student
        for role in ["teacher", "student"]:
            if role in self.config:
                provider_type = self.config[role]["provider"]
                
                if provider_type == "modal":
                    provider_config = self.config.get("providers", {}).get("modal", {})
                    self.model_providers[role] = ProviderFactory.create_model_provider("modal", provider_config)
                    
                elif provider_type == "local":
                    provider_config = self.config.get("providers", {}).get("local", {})
                    if "ollama" in self.config:
                        provider_config.update(self.config["ollama"])
                    self.model_providers[role] = ProviderFactory.create_model_provider("local", provider_config)
                    
                elif provider_type in ["anthropic", "openai"]:
                    # These will be handled directly by DSPy LM
                    self.model_providers[role] = None
                else:
                    raise ValueError(f"Unsupported provider for {role}: {provider_type}")
        
        # Initialize artifact provider
        artifact_config = self.config.get("artifact_storage", {})
        artifact_type = artifact_config.get("type", "modal")
        
        if artifact_type == "modal":
            provider_config = artifact_config.get("modal", {})
            self.artifact_providers["primary"] = ProviderFactory.create_artifact_provider("modal", provider_config)
        elif artifact_type == "local":
            provider_config = artifact_config.get("local", {})
            self.artifact_providers["primary"] = ProviderFactory.create_artifact_provider("local", provider_config)
        else:
            raise ValueError(f"Unsupported artifact provider: {artifact_type}")
    
    def deploy_hosted_models(self) -> Dict[str, str]:
        """Deploy hosted models if needed and return their URLs."""
        hosted_models = []
        
        # Check which models need hosted deployment
        for role in ["teacher", "student"]:
            if role in self.config and self.config[role]["provider"] in ["modal", "local"]:
                hosted_models.append((role, self.config[role]["model"]))
        
        if not hosted_models:
            print("📋 No hosted models to deploy")
            return {}
        
        print(f"🚀 Deploying {len(hosted_models)} hosted models...")
        
        deployed_services = {}
        
        for role, model_id in hosted_models:
            if role in self.model_providers and self.model_providers[role]:
                try:
                    services = self.model_providers[role].deploy_model_service(model_id)
                    deployed_services[role] = services
                    print(f"✅ {role.title()} model deployed successfully")
                except Exception as e:
                    print(f"❌ Failed to deploy {role} model: {e}")
                    raise
        
        return deployed_services
    
    def call_model(
        self,
        model_type: str,  # "teacher" or "student"
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 150
    ) -> str:
        """
        Call a model through its provider.
        """
        model_config = self.config[model_type]
        model_id = model_config["model"]
        provider_type = model_config["provider"]
        
        print(f"🤖 Calling {model_type} model: {model_id} via {provider_type}")
        
        if provider_type in ["modal", "local"] and model_type in self.model_providers:
            # Use provider
            provider = self.model_providers[model_type]
            if provider:
                return provider.call_model(model_id, prompt, system_prompt, temperature, max_tokens)
        
        # Fallback to direct API calls for anthropic/openai
        return self._call_api_model(model_id, provider_type, prompt, system_prompt, temperature, max_tokens)
    
    def _call_api_model(self, model_id: str, provider_type: str, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
        """Call API-based models directly."""
        print(f"\n🌐 Calling API model: {model_id} via {provider_type}")
        print(f"   Prompt: {prompt[:100]}...")
        
        if provider_type == "anthropic":
            import anthropic
            api_key = os.getenv(self.config["providers"]["anthropic"]["api_key_env"])
            if not api_key:
                raise Exception("Anthropic API key not found")
            
            client = anthropic.Anthropic(api_key=api_key)
            messages = [{"role": "user", "content": prompt}]
            
            response = client.messages.create(
                model=model_id,
                system=system_prompt if system_prompt else None,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.content[0].text
            
        elif provider_type == "openai":
            import openai
            api_key = os.getenv(self.config["providers"]["openai"]["api_key_env"])
            if not api_key:
                raise Exception("OpenAI API key not found")
            
            client = openai.OpenAI(api_key=api_key)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")
    
    def upload_artifacts(self, local_path: str, remote_path: str = "/artifacts/unified_router_prompt_artifact.json") -> bool:
        """Upload artifacts using the configured provider."""
        if "primary" in self.artifact_providers:
            return self.artifact_providers["primary"].upload_artifact(local_path, remote_path)
        return False


# ==================== DSPy Integration ====================

class ModelClientLM(DSPyLMProvider):
    """DSPy LM wrapper that uses the ModelClient."""
    
    def __init__(self, client: ModelClient, model_type: str):
        # Create a provider instance for DSPy
        if model_type in client.model_providers and client.model_providers[model_type]:
            super().__init__(client.model_providers[model_type], client.config[model_type]["model"], model_type)
        else:
            # For API-based models, we'll use a wrapper
            super().__init__(None, client.config[model_type]["model"], model_type)
            
        self.client = client
        self.model_type = model_type
        
    def basic_generate(self, prompt, **kwargs):
        """Generate using the client."""
        if self.provider:
            # Use provider
            return super().basic_generate(prompt, **kwargs)
        else:
            # Use client for API-based models
            temperature = kwargs.get("temperature", 0.1)
            max_tokens = kwargs.get("max_tokens", 150)
            
            response = self.client.call_model(
                model_type=self.model_type,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return [response]


# ==================== Main Orchestrator ====================

class OptimizationOrchestrator:
    """Main orchestrator for the optimization process."""
    
    def __init__(self, config_path: str = "config.json"):
        
        self.config_instance = get_config()
        self.config = self._load_config()
        self.client = ModelClient(self.config)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration using Config class."""
        # Get full config from Config instance
        full_config = self.config_instance.get_all()
        
        # Extract optimization config
        if "optimization" not in full_config:
            raise ValueError("No 'optimization' block found in config.json")
        
        opt_config = full_config["optimization"]
        
        # Check if optimization is enabled
        if not opt_config.get("enabled", False):
            raise ValueError("Optimization is disabled in config. Set 'optimization.enabled' to true.")
        
        # Check optimization type
        if opt_config.get("type") != "dspy":
            raise ValueError(f"Unsupported optimization type: {opt_config.get('type')}. Only 'dspy' is supported.")
        
        print(f"📋 Loaded configuration from {self.config_path}")
        print(f"   Teacher: {opt_config['teacher']['model']} ({opt_config['teacher']['provider']})")
        print(f"   Student: {opt_config['student']['model']} ({opt_config['student']['provider']})")
        
        return opt_config
    
    def setup_services(self):
        """Deploy any needed hosted services."""
        print("\n🛠️ Setting up services...")
        
        # Deploy hosted models if needed
        print("📋 Calling deploy_hosted_models...")
        deployed_services = self.client.deploy_hosted_models()
        print(f"✅ Deployed services: {deployed_services}")
        
        # Test connections
        print("🧪 Testing connections...")
        self._test_provider_connections()
        print("✅ Connection tests complete")
    
    def _test_provider_connections(self):
        """Test provider connections."""
        print("\n🧪 Testing provider connections...")
        
        for role in ["teacher", "student"]:
            if role in self.client.model_providers:
                provider = self.client.model_providers[role]
                if provider:
                    try:
                        health = provider.health_check()
                        status = health.get("status", "unknown")
                        print(f"   {role.title()}: {status}")
                    except Exception as e:
                        print(f"   {role.title()}: Error - {e}")
    
    def generate_training_data(self) -> List:
        """Generate training data using the teacher model."""
        print("\n📚 Generating training data...")
        
        # Import here to avoid circular imports
        import dspy
        
        # Check if we have cached training data
        cache_file = Path("optimization_results/teacher_training_cache.json")
        if cache_file.exists():
            print("📦 Loading cached training data...")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            print(f"✅ Loaded {len(cached_data)} cached examples")
            
            # Convert back to Example objects
            examples = []
            for item in cached_data:
                example = dspy.Example(
                    conversation_history=item["conversation_history"],
                    user_query=item["user_query"],
                    routing_decision=RoutingDecision(**item["routing_decision"])
                ).with_inputs("conversation_history", "user_query")
                examples.append(example)
            return examples
        
        # Generate diverse queries
        queries = self._generate_diverse_queries()
        print(f"📝 Generated {len(queries)} diverse queries")
        
        # Use teacher to label examples
        examples = []
        print("🤖 Creating teacher LM...")
        
        # Import here to avoid circular imports
        import dspy
        
        # Get teacher config
        teacher_config = self.client.config["teacher"]
        teacher_model = teacher_config["model"]
        teacher_provider = teacher_config["provider"]
        
        # Create DSPy LM using LiteLLM format
        if teacher_provider == "anthropic":
            import os
            api_key = os.getenv(self.client.config["providers"]["anthropic"]["api_key_env"])
            # Use LiteLLM format for Anthropic
            teacher_lm = dspy.LM(
                model="claude-3-5-sonnet-20241022",  # LiteLLM handles this
                api_key=api_key,
                temperature=0.7
            )
        else:
            # For other providers, use standard format
            teacher_lm = dspy.LM(model=teacher_model, temperature=0.7)
        
        # Configure DSPy with teacher
        print("🔧 Configuring DSPy with teacher LM...")
        with dspy.context(lm=teacher_lm):
            print("📚 Creating teacher router module...")
            teacher_router = RouterModule()
            
            print(f"🎯 Processing {len(queries)} queries with teacher...")
            for i, query_data in enumerate(queries):
                try:
                    if isinstance(query_data, dict):
                        conversation_history = query_data.get("conversation_history", "")
                        user_query = query_data["user_query"]
                    else:
                        conversation_history = ""
                        user_query = query_data
                    
                    print(f"\n   [{i+1}/{len(queries)}] Query: {user_query[:50]}...")
                    
                    # Get routing decision from teacher
                    print("   🤔 Calling teacher model...")
                    decision = teacher_router(
                        conversation_history=conversation_history,
                        user_query=user_query
                    )
                    
                    examples.append(dspy.Example(
                        conversation_history=conversation_history,
                        user_query=user_query,
                        routing_decision=decision
                    ).with_inputs("conversation_history", "user_query"))
                    
                except Exception as e:
                    print(f"   ❌ Error generating example {i+1}: {e}")
                    continue
        
        print(f"✅ Generated {len(examples)} training examples")
        
        # Save the training examples to cache
        from cogniverse_core.common.utils.output_manager import get_output_manager
        output_manager = get_output_manager()
        cache_dir = output_manager.get_optimization_dir()
        cache_file = cache_dir / "teacher_training_cache.json"
        
        cache_data = []
        for example in examples:
            cache_data.append({
                "conversation_history": example.conversation_history,
                "user_query": example.user_query,
                "routing_decision": {
                    "search_modality": example.routing_decision.search_modality,
                    "generation_type": example.routing_decision.generation_type
                }
            })
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"💾 Saved {len(cache_data)} examples to cache: {cache_file}")
        
        return examples
    
    def _generate_diverse_queries(self) -> List:
        """Generate diverse training queries."""
        templates = {
            "video_raw": [
                "Show me how to {action}",
                "Find tutorials on {topic}",
                "I want to watch videos about {subject}",
                "Play the {video_name} video"
            ],
            "text_report": [
                "Create a detailed report on {topic}",
                "Construct a comprehensive analysis of {subject}",
                "Build an in-depth report about {area}"
            ],
            "text_summary": [
                "What is the main point of {document}?",
                "Summarize the key findings about {topic}",
                "Give me a quick overview of {subject}"
            ]
        }
        
        topics = [
            "artificial intelligence", "climate change", "quantum computing",
            "renewable energy", "space exploration", "biotechnology"
        ]
        
        actions = [
            "cook pasta", "fix a bicycle", "learn guitar", "code in Python"
        ]
        
        queries = []
        import random
        
        num_examples = self.config["settings"]["num_examples"]
        queries_per_template = num_examples // len(templates)
        
        for category, template_list in templates.items():
            for _ in range(queries_per_template):
                template = random.choice(template_list)
                query = template.format(
                    action=random.choice(actions),
                    topic=random.choice(topics),
                    subject=random.choice(topics),
                    area=random.choice(topics),
                    video_name=f"tutorial on {random.choice(topics)}",
                    document=f"the paper on {random.choice(topics)}"
                )
                
                if random.random() > 0.7:
                    queries.append({
                        "conversation_history": f"User previously asked about {random.choice(topics)}",
                        "user_query": query
                    })
                else:
                    queries.append(query)
        
        return queries
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run the full optimization process using the existing optimizer."""
        print("\n🚀 Starting Agentic Router Optimization")
        print("=" * 60)
        
        # 1. Setup services
        self.setup_services()
        
        # 2. Run optimization using existing logic
        return self._run_provider_optimization(
            student_model=self.config["student"]["model"],
            teacher_model=self.config["teacher"]["model"],
            num_teacher_examples=self.config["settings"]["num_examples"],
            output_dir=self.config["output"]["dir"]
        )
    
    def _run_provider_optimization(
        self,
        student_model: str,
        teacher_model: str,
        num_teacher_examples: int,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Run optimization using the existing optimizer logic but with provider API calls.
        """
        import dspy
        from dspy.teleprompt import MIPROv2
        
        start_time = time.time()
        
        # 1. Generate training data using teacher
        print("\n📚 Generating training data with teacher model...")
        training_examples = self.generate_training_data()
        
        # Split into train/val
        split_idx = int(0.8 * len(training_examples))
        train_set = training_examples[:split_idx]
        val_set = training_examples[split_idx:]
        
        print(f"📊 Dataset: {len(train_set)} train, {len(val_set)} validation")
        
        # 2. Configure DSPy with student
        student_config = self.client.config["student"]
        student_model = student_config["model"]
        student_provider = student_config["provider"]
        
        # Create student LM
        print(f"\n🎓 Configuring student model: {student_model} via {student_provider}")
        if student_provider == "modal":
            # For Modal, we need to use the deployed endpoint
            # Use Config class to get configuration
            config = get_config()
            modal_endpoint = config.get("inference.modal_endpoint")
            
            print(f"📍 Modal endpoint from config: {modal_endpoint}")
            
            if modal_endpoint:
                # Use custom provider format for Modal
                print("🔧 Creating DSPy LM with Modal endpoint...")
                # Modal endpoint should be the vLLM serve endpoint
                print(f"📍 Using Modal endpoint: {modal_endpoint}")
                
                # The vLLM serve endpoint already provides OpenAI-compatible API
                # Just use it directly as the base URL
                
                # Use OpenAI format with the vLLM endpoint
                # For custom endpoints, we need to prefix with "openai/" to tell LiteLLM to use OpenAI provider
                # Ensure the endpoint includes /v1 for OpenAI compatibility
                if not modal_endpoint.endswith('/v1'):
                    modal_endpoint = modal_endpoint.rstrip('/') + '/v1'
                
                student_lm = dspy.LM(
                    model=f"openai/{student_model}",  # Prefix with openai/ for custom endpoints
                    api_base=modal_endpoint,
                    api_key="dummy",  # vLLM doesn't need auth
                    temperature=0.3,  # Lower temp for Gemma (better at following instructions)
                    max_tokens=2048,  # Reasonable for routing decisions
                    timeout=300  # 5 minute timeout per request
                )
                print("✅ Student LM created")
            else:
                raise ValueError("Modal endpoint not found for student model")
        else:
            # For other providers
            print(f"🔧 Creating DSPy LM for {student_provider}...")
            student_lm = dspy.LM(model=student_model, temperature=0.1, max_tokens=100)
            print("✅ Student LM created")
        
        print("🔧 Configuring DSPy with student LM...")
        dspy.configure(lm=student_lm)
        print("✅ DSPy configured")
        
        # 3. Create router module (using existing class)
        print("\n🤖 Creating router module...")
        router = RouterModule()
        print("✅ Router module created")
        
        # 4. Baseline evaluation
        print("\n📈 Evaluating baseline performance...")
        print(f"   Validation set size: {len(val_set)}")
        baseline_metrics = evaluate_routing_accuracy(router, val_set)
        print(f"Baseline accuracy: {baseline_metrics['overall_accuracy']:.2%}")
        
        # 5. Run MIPROv2 optimization
        print("\n🔧 Running MIPROv2 optimization...")
        
        def routing_metric(example, prediction) -> float:
            score = 0.0
            if not isinstance(prediction, RoutingDecision):
                return 0.0
            
            if prediction.search_modality == example.routing_decision.search_modality:
                score += 0.5
            if prediction.generation_type == example.routing_decision.generation_type:
                score += 0.5
            
            return score
        
        optimizer = MIPROv2(
            metric=routing_metric,
            init_temperature=0.7,
            verbose=True,
            auto="light",  # Use auto mode instead of manual settings
            num_threads=8  # Enable parallel generation with 8 threads
        )
        
        optimized_router = optimizer.compile(
            router,
            trainset=train_set,
            valset=val_set,
            # Don't specify num_trials when using auto mode
            minibatch_size=4,
            minibatch_full_eval_steps=10,
            minibatch=True,
            requires_permission_to_run=False
        )
        
        # 6. Evaluate optimized performance
        print("\n📊 Evaluating optimized performance...")
        optimized_metrics = evaluate_routing_accuracy(optimized_router, val_set)
        print(f"Optimized accuracy: {optimized_metrics['overall_accuracy']:.2%}")
        
        optimization_time = time.time() - start_time
        
        # 7. Extract and save artifacts (using existing logic)
        state = optimized_router.dump_state()
        
        artifacts = {
            "instructions": state.get("route", {}).get("instructions", ""),
            "demonstrations": [],
            "model_config": {
                "student_model": student_model,
                "teacher_model": teacher_model,
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
        
        # Save results
        from cogniverse_core.common.utils.output_manager import get_output_manager
        output_manager = get_output_manager()
        output_path = output_manager.get_optimization_dir()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_path / f"unified_optimization_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(artifacts, f, indent=2)
        
        # Save integration artifact
        integration_file = output_path / "unified_router_prompt_artifact.json"
        integration_data = {
            "system_prompt": artifacts["instructions"],
            "few_shot_examples": artifacts["demonstrations"],
            "model_config": artifacts["model_config"]
        }
        with open(integration_file, "w") as f:
            json.dump(integration_data, f, indent=2)
        
        # Upload to provider (Modal volume or local filesystem)
        self.client.upload_artifacts(str(integration_file))
        
        print(f"💾 Results saved to: {output_file}")
        print(f"💾 Integration artifact: {integration_file}")
        print("☁️ Artifacts uploaded via provider")
        
        return artifacts


# ==================== CLI Interface ====================

def main():
    """Main orchestrator function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic Router Optimization Orchestrator")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--setup-only", action="store_true", help="Only setup services, don't run optimization")
    parser.add_argument("--test-models", action="store_true", help="Test model connections")
    
    args = parser.parse_args()
    
    try:
        orchestrator = OptimizationOrchestrator(args.config)
        
        if args.setup_only:
            orchestrator.setup_services()
            print("✅ Services setup complete")
        elif args.test_models:
            orchestrator.setup_services()
            print("✅ Model tests complete")
        else:
            result = orchestrator.run_optimization()
            
            print("\n" + "=" * 60)
            print("✨ OPTIMIZATION COMPLETE")
            print("=" * 60)
            print(f"Time: {result['optimization_time']:.2f} seconds")
            print(f"Improvement: {result['metrics']['improvement']['overall']:.2%}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
