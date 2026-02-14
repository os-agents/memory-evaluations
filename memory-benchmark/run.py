"""
Main entry point for running memory provider evaluations
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
ENV_PATH = find_dotenv(usecwd=True)
if ENV_PATH:
    load_dotenv(ENV_PATH)

# Import evaluator
from eval.evaluate import MemoryEvaluator


def load_provider_config(provider_name: str) -> Dict[str, Any]:
    """
    Load provider-specific tuning config from configs/providers/<provider>.json.
    Returns {} when no config file exists.
    """
    config_path = Path("configs") / "providers" / f"{provider_name}.json"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load config for {provider_name} from {config_path}: {e}")
        return {}


def setup_environment(provider_names=None):
    """
    Check environment and setup
    """
    print("🔍 Checking environment configuration...\n")
    
    # Check for .env file
    if not ENV_PATH:
        print("⚠️  No .env file found!")
        print("   Please create a .env file from .env.example:")
        print("   cp .env.example .env")
        print("   Then add your API keys to the .env file\n")
        return False
    
    provider_names = provider_names or ["mem0"]

    # Check for required API keys
    missing_keys = []

    required_env_by_provider = {
        "mem0": ["MEM0_API_KEY"],
        "zep": ["ZEP_API_KEY"],
    }

    for provider_name in provider_names:
        for env_key in required_env_by_provider.get(provider_name, []):
            if not os.getenv(env_key):
                missing_keys.append(env_key)

    missing_keys = sorted(set(missing_keys))
    
    if missing_keys:
        print(f"⚠️  Missing API keys in .env file: {', '.join(missing_keys)}")
        print("   Please add them to your .env file\n")
        return False
    
    print("✅ Environment configured successfully\n")
    return True


def run_evaluation(provider_name: str = "mem0", save_results: bool = True, provider_config: Optional[Dict[str, Any]] = None):
    """
    Run evaluation for a specific provider
    
    Args:
        provider_name: Name of provider to evaluate
        save_results: Whether to save results to file
    """
    print(f"\n🚀 Starting Memory Provider Evaluation")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize provider
    print(f"🔧 Initializing provider: {provider_name}")
    provider_config = provider_config or {}
    
    try:
        if provider_name == "mem0":
            from providers.mem0 import Mem0Provider
            provider = Mem0Provider()
        elif provider_name == "zep":
            from providers.zep import ZepProvider
            provider = ZepProvider(config=provider_config)
        else:
            print(f"✗ Unknown provider: {provider_name}")
            return None
        print(f"✓ {provider_name} initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize {provider_name}: {e}")
        return None
    
    # Initialize evaluator
    print(f"\n📋 Loading scenarios from scenarios.json")
    try:
        evaluator = MemoryEvaluator("datasets/scenarios.json")
        print(f"✓ Loaded {len(evaluator.scenarios)} scenarios")
    except Exception as e:
        print(f"✗ Failed to load scenarios: {e}")
        return None
    
    # Run evaluation
    print(f"\n🏃 Running evaluation...")
    try:
        results = evaluator.run_evaluation(provider)
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Attach run metadata for reproducibility.
    results["provider_config"] = provider_config
    
    # Generate report
    print(f"\n📊 Generating report...")
    report = evaluator.generate_report(results)
    print("\n" + report)
    
    # Save results
    if save_results:
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = results_dir / f"{provider_name}_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {json_path}")
        
        # Save report
        report_path = results_dir / f"{provider_name}_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"💾 Report saved to: {report_path}")
    
    return results


def compare_providers(provider_names: list):
    """
    Run evaluation for multiple providers and compare
    
    Args:
        provider_names: List of provider names to compare
    """
    all_results = {}
    
    for provider_name in provider_names:
        print(f"\n{'='*80}")
        print(f"Evaluating: {provider_name}")
        print(f"{'='*80}")
        
        provider_config = load_provider_config(provider_name)
        results = run_evaluation(provider_name, save_results=True, provider_config=provider_config)
        if results:
            all_results[provider_name] = results
    
    # Generate comparison report
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        # Compare overall metrics
        print(f"{'Provider':<20} {'Precision':<12} {'Recall':<12} {'Exact Match':<12} {'Latency (ms)':<12}")
        print("-" * 80)
        
        for provider_name, results in all_results.items():
            overall = results['aggregate_metrics']['overall']
            print(
                f"{provider_name:<20} "
                f"{overall['avg_precision_at_k']:<12.2%} "
                f"{overall['avg_recall_at_k']:<12.2%} "
                f"{overall['exact_match_rate']:<12.2%} "
                f"{overall['avg_latency_ms']:<12.2f}"
            )
        
        # Save comparison
        results_dir = Path("results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = results_dir / f"comparison_{timestamp}.json"
        
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Comparison saved to: {comparison_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Evaluate memory providers for AI agents"
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        default=None,
        help='Provider to evaluate (supported: mem0, zep). If omitted, runs mem0 and zep.'
    )
    
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Compare multiple providers (e.g., --compare mem0 zep)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Don\'t save results to file'
    )
    
    args = parser.parse_args()
    
    # Setup environment
    provider_names = args.compare if args.compare else ([args.provider] if args.provider else ["mem0", "zep"])
    if not setup_environment(provider_names=provider_names):
        print("\n⚠️  Environment setup incomplete. Some evaluations may fail.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Run evaluation(s)
    if args.compare:
        compare_providers(args.compare)
    elif args.provider:
        provider_config = load_provider_config(args.provider)
        run_evaluation(args.provider, save_results=not args.no_save, provider_config=provider_config)
    else:
        for provider_name in ["mem0", "zep"]:
            print(f"\n{'='*80}")
            print(f"Evaluating: {provider_name}")
            print(f"{'='*80}")
            provider_config = load_provider_config(provider_name)
            run_evaluation(provider_name, save_results=not args.no_save, provider_config=provider_config)
    
    print("\n✅ Evaluation complete!\n")


if __name__ == "__main__":
    main()

