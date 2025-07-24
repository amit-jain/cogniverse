#!/usr/bin/env python3
"""
Comprehensive Analysis of GLiNER Optimization Approaches

This analyzes the results from different optimization strategies:
1. Manual Grid Search (original gliner_optimizer.py)
2. Simple Strategic Optimization (simple_dspy_optimizer.py)
3. DSPy-based Optimization (dspy_gliner_optimizer.py)
"""

import json
import os
from typing import Dict, Any, List

class OptimizationAnalysis:
    """Analyze and compare different GLiNER optimization approaches."""
    
    def __init__(self):
        self.results = {}
        self.load_results()
    
    def load_results(self):
        """Load results from different optimization runs."""
        
        # Load simple optimizer results
        simple_file = "simple_gliner_optimization_results.json"
        if os.path.exists(simple_file):
            with open(simple_file, 'r') as f:
                self.results["simple"] = json.load(f)
        
        # Load manual optimizer results (if available)
        manual_file = "gliner_optimization_results.json"
        if os.path.exists(manual_file):
            with open(manual_file, 'r') as f:
                self.results["manual"] = json.load(f)
        
        # Load DSPy optimizer results (if available)
        dspy_file = "dspy_gliner_optimization_results.json"
        if os.path.exists(dspy_file):
            with open(dspy_file, 'r') as f:
                self.results["dspy"] = json.load(f)
    
    def analyze_performance(self):
        """Analyze performance across different optimization approaches."""
        
        print("🔍 GLiNER Optimization Performance Analysis")
        print("=" * 60)
        
        # Performance summary
        performance_summary = {}
        
        for approach_name, approach_results in self.results.items():
            if approach_name == "simple":
                # Simple optimizer format
                accuracies = [result["accuracy"] for result in approach_results.values()]
                eval_times = [result["eval_time"] for result in approach_results.values()]
                
                performance_summary[approach_name] = {
                    "best_accuracy": max(accuracies) if accuracies else 0,
                    "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                    "best_speed": min(eval_times) if eval_times else 0,
                    "avg_speed": sum(eval_times) / len(eval_times) if eval_times else 0,
                    "configs_tested": len(approach_results),
                    "approach": "Strategic Configuration Testing"
                }
                
            elif approach_name == "manual":
                # Manual optimizer format (list of results)
                if isinstance(approach_results, list):
                    accuracies = [result["accuracy"] for result in approach_results]
                    runtimes = [result["runtime"] for result in approach_results]
                    
                    performance_summary[approach_name] = {
                        "best_accuracy": max(accuracies) if accuracies else 0,
                        "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                        "best_speed": min(runtimes) if runtimes else 0,
                        "avg_speed": sum(runtimes) / len(runtimes) if runtimes else 0,
                        "configs_tested": sum(result.get("details", {}).get("configurations_tested", 0) for result in approach_results),
                        "approach": "Manual Grid Search + Adaptive"
                    }
                    
            elif approach_name == "dspy":
                # DSPy optimizer format
                all_accuracies = []
                all_times = []
                total_configs = 0
                
                for optimizer_type, optimizer_results in approach_results.items():
                    if isinstance(optimizer_results, dict):
                        for model_name, model_results in optimizer_results.items():
                            if isinstance(model_results, dict) and "accuracy" in model_results:
                                all_accuracies.append(model_results["accuracy"])
                                all_times.append(model_results["optimization_time"])
                                total_configs += 1
                
                if all_accuracies:
                    performance_summary[approach_name] = {
                        "best_accuracy": max(all_accuracies),
                        "avg_accuracy": sum(all_accuracies) / len(all_accuracies),
                        "best_speed": min(all_times) if all_times else 0,
                        "avg_speed": sum(all_times) / len(all_times) if all_times else 0,
                        "configs_tested": total_configs,
                        "approach": "DSPy Automatic Optimization"
                    }
        
        # Print performance comparison
        print("\n📊 Performance Comparison:")
        print(f"{'Approach':<20} {'Best Acc':<10} {'Avg Acc':<10} {'Best Time':<12} {'Configs':<10}")
        print("-" * 70)
        
        for approach_name, stats in performance_summary.items():
            print(f"{approach_name:<20} {stats['best_accuracy']:<10.1%} {stats['avg_accuracy']:<10.1%} "
                  f"{stats['best_speed']:<12.2f}s {stats['configs_tested']:<10d}")
        
        # Find the best approach
        best_approach = max(performance_summary.items(), key=lambda x: x[1]["best_accuracy"])
        print(f"\n🏆 Best Approach: {best_approach[0]} ({best_approach[1]['best_accuracy']:.1%})")
        
        return performance_summary
    
    def analyze_configuration_insights(self):
        """Analyze insights from different configurations."""
        
        print("\n🔍 Configuration Insights:")
        print("-" * 40)
        
        if "simple" in self.results:
            simple_results = self.results["simple"]
            
            # Sort by accuracy
            sorted_configs = sorted(simple_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
            
            print("\n📋 Top Configurations (Simple Optimizer):")
            for i, (config_name, result) in enumerate(sorted_configs[:3], 1):
                config = result["config"]
                print(f"{i}. {config_name}: {result['accuracy']:.1%}")
                print(f"   Model: {config['model'].split('/')[-1]}")
                print(f"   Labels: {len(config['labels'])} labels")
                print(f"   Threshold: {config['threshold']}")
                print(f"   Speed: {result['eval_time']:.2f}s")
                print()
            
            # Performance vs complexity analysis
            print("🎯 Performance vs Complexity Analysis:")
            
            # Model size analysis
            model_performance = {}
            for config_name, result in simple_results.items():
                model = result["config"]["model"].split('/')[-1]
                if model not in model_performance:
                    model_performance[model] = []
                model_performance[model].append(result["accuracy"])
            
            for model, accuracies in model_performance.items():
                avg_acc = sum(accuracies) / len(accuracies)
                print(f"   {model}: {avg_acc:.1%} average accuracy")
            
            # Label count analysis
            label_performance = {}
            for config_name, result in simple_results.items():
                label_count = len(result["config"]["labels"])
                label_performance[label_count] = label_performance.get(label_count, [])
                label_performance[label_count].append(result["accuracy"])
            
            print("\n📊 Label Count Impact:")
            for label_count, accuracies in sorted(label_performance.items()):
                avg_acc = sum(accuracies) / len(accuracies)
                print(f"   {label_count} labels: {avg_acc:.1%} average accuracy")
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis."""
        
        print("\n💡 Optimization Recommendations:")
        print("=" * 40)
        
        if "simple" in self.results:
            simple_results = self.results["simple"]
            
            # Find best performing config
            best_config = max(simple_results.items(), key=lambda x: x[1]["accuracy"])
            best_name, best_result = best_config
            
            print(f"🥇 Recommended Configuration: {best_name}")
            print(f"   📈 Expected Accuracy: {best_result['accuracy']:.1%}")
            print(f"   🤖 Model: {best_result['config']['model']}")
            print(f"   🏷️ Labels: {best_result['config']['labels']}")
            print(f"   🎯 Threshold: {best_result['config']['threshold']}")
            print(f"   ⏱️ Evaluation Time: {best_result['eval_time']:.2f}s")
            
            # Speed vs accuracy tradeoff
            fastest_config = min(simple_results.items(), key=lambda x: x[1]["eval_time"])
            fastest_name, fastest_result = fastest_config
            
            print(f"\n⚡ Fastest Configuration: {fastest_name}")
            print(f"   📈 Accuracy: {fastest_result['accuracy']:.1%}")
            print(f"   ⏱️ Evaluation Time: {fastest_result['eval_time']:.2f}s")
            print(f"   🔄 Speed vs Best: {fastest_result['eval_time'] / best_result['eval_time']:.1f}x faster")
            
            # Balanced recommendation
            balanced_scores = {}
            for config_name, result in simple_results.items():
                # Balance accuracy and speed (normalized)
                acc_score = result["accuracy"]
                speed_score = 1 / (result["eval_time"] + 0.1)  # Invert and add small constant
                balanced_scores[config_name] = (acc_score * 0.7 + speed_score * 0.3, result)
            
            balanced_best = max(balanced_scores.items(), key=lambda x: x[1][0])
            balanced_name, (balanced_score, balanced_result) = balanced_best
            
            print(f"\n⚖️ Balanced Recommendation: {balanced_name}")
            print(f"   📈 Accuracy: {balanced_result['accuracy']:.1%}")
            print(f"   ⏱️ Evaluation Time: {balanced_result['eval_time']:.2f}s")
            print(f"   🎯 Balanced Score: {balanced_score:.3f}")
        
        print(f"\n🚀 Next Steps:")
        print("1. Use the recommended configuration in production")
        print("2. Monitor performance on real queries")
        print("3. Consider implementing DSPy for automatic optimization")
        print("4. Collect more training data for better evaluation")
        print("5. Test with domain-specific query patterns")

def main():
    """Run comprehensive optimization analysis."""
    
    analyzer = OptimizationAnalysis()
    
    if not analyzer.results:
        print("❌ No optimization results found!")
        print("Run the optimizers first:")
        print("  python tests/routing/simple_dspy_optimizer.py")
        print("  python tests/routing/gliner_optimizer.py")
        return
    
    print(f"📊 Analyzing {len(analyzer.results)} optimization approaches")
    
    # Performance analysis
    performance = analyzer.analyze_performance()
    
    # Configuration insights
    analyzer.analyze_configuration_insights()
    
    # Recommendations
    analyzer.generate_recommendations()
    
    print(f"\n📝 Analysis Summary:")
    print("=" * 40)
    
    if performance:
        best_approach = max(performance.items(), key=lambda x: x[1]["best_accuracy"])
        print(f"🏆 Best Overall: {best_approach[0]} ({best_approach[1]['best_accuracy']:.1%})")
        print(f"🎯 Key Insight: {best_approach[1]['approach']}")
        
        # Efficiency analysis
        efficiency_scores = {}
        for name, stats in performance.items():
            if stats["best_speed"] > 0:
                efficiency = stats["best_accuracy"] / stats["best_speed"]
                efficiency_scores[name] = efficiency
        
        if efficiency_scores:
            most_efficient = max(efficiency_scores.items(), key=lambda x: x[1])
            print(f"⚡ Most Efficient: {most_efficient[0]} ({most_efficient[1]:.3f} acc/sec)")
    
    print(f"\n🔄 Optimization Approach Comparison:")
    print("• Simple Strategic: Fast, targeted testing of promising configurations")
    print("• Manual Grid Search: Comprehensive but slow, tests all combinations")
    print("• DSPy Automatic: Intelligent optimization with learning capabilities")
    print("")
    print("💡 Recommendation: Use Simple Strategic for quick improvements,")
    print("   DSPy for long-term optimization and learning.")

if __name__ == "__main__":
    main()