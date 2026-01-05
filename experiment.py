"""
Experiment runner to compare compression methods.
"""

from compressors import create_compressor, Compressor
from llm_client import create_client, LLMClient
import os
import json
import time
from typing import List, Dict, Any
from datetime import datetime

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CompressionExperiment:
    """Run experiments comparing different compression methods."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize experiment runner.

        Args:
            llm_client: LLM client to use for all experiments
        """
        self.client = llm_client

    def run_single_test(
        self,
        prompt: str,
        compressor: Compressor,
        method_name: str,
        task_description: str = ""
    ) -> Dict[str, Any]:
        """
        Run a single test with one compression method.

        Args:
            prompt: Original prompt
            compressor: Compressor to use
            method_name: Name of the method for logging
            task_description: Optional description of the task

        Returns:
            Dictionary with test results
        """
        # Compress the prompt
        start_time = time.time()
        compressed_prompt = compressor.compress(prompt)
        compression_time = time.time() - start_time

        # Count tokens
        original_tokens = self.client.count_tokens(prompt)
        compressed_tokens = self.client.count_tokens(compressed_prompt)

        # Send to LLM
        response, token_stats = self.client.send_prompt(compressed_prompt)

        # Calculate metrics
        compression_ratio = len(compressed_prompt) / \
            len(prompt) if len(prompt) > 0 else 1.0
        token_reduction = original_tokens - compressed_tokens
        token_reduction_pct = (
            token_reduction / original_tokens * 100) if original_tokens > 0 else 0

        return {
            "method": method_name,
            "task_description": task_description,
            "original_prompt": prompt,
            "compressed_prompt": compressed_prompt,
            "response": response,
            "original_length": len(prompt),
            "compressed_length": len(compressed_prompt),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "token_reduction": token_reduction,
            "token_reduction_pct": token_reduction_pct,
            "compression_ratio": compression_ratio,
            "compression_time_sec": compression_time,
            "input_tokens_used": token_stats["input_tokens"],
            "output_tokens_used": token_stats["output_tokens"],
            "total_tokens_used": token_stats["total_tokens"],
        }

    def run_comparison(
        self,
        prompts: List[Dict[str, str]],
        compression_methods: List[str] = None,
        output_file: str = None
    ) -> List[Dict[str, Any]]:
        """
        Run comparison across multiple prompts and compression methods.

        Args:
            prompts: List of dicts with 'prompt' and 'task_description' keys
            compression_methods: List of method names ('none', 'caveman', 'llmlingua')
            output_file: Optional file to save results as JSON

        Returns:
            List of all test results
        """
        if compression_methods is None:
            compression_methods = ['none', 'caveman', 'llmlingua']

        all_results = []

        print(f"\n{'='*80}")
        print(f"Starting Compression Experiment")
        print(f"{'='*80}")
        print(f"Number of test prompts: {len(prompts)}")
        print(f"Compression methods: {', '.join(compression_methods)}")
        print(f"LLM Client: {self.client.__class__.__name__}")
        print(f"{'='*80}\n")

        for idx, prompt_data in enumerate(prompts):
            prompt = prompt_data['prompt']
            task_desc = prompt_data.get('task_description', f'Task {idx + 1}')

            print(f"\n--- Test {idx + 1}/{len(prompts)}: {task_desc} ---")

            for method in compression_methods:
                print(f"\nTesting method: {method}")

                # Create compressor
                if method == 'llmlingua':
                    compressor = create_compressor(method, rate=0.5)
                else:
                    compressor = create_compressor(method)

                # Run test
                try:
                    result = self.run_single_test(
                        prompt=prompt,
                        compressor=compressor,
                        method_name=method,
                        task_description=task_desc
                    )
                    all_results.append(result)

                    # Print summary
                    print(f"  Original tokens: {result['original_tokens']}")
                    print(
                        f"  Compressed tokens: {result['compressed_tokens']}")
                    print(
                        f"  Token reduction: {result['token_reduction_pct']:.1f}%")
                    print(
                        f"  Response length: {len(result['response'])} chars")

                except Exception as e:
                    print(f"  ERROR: {str(e)}")
                    continue

        # Print summary statistics
        self._print_summary(all_results)

        # Save results if requested
        if output_file:
            self._save_results(all_results, output_file)

        return all_results

    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*80}\n")

        # Group by method
        by_method = {}
        for result in results:
            method = result['method']
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)

        # Calculate averages
        for method, method_results in by_method.items():
            print(f"\n{method.upper()} METHOD:")
            print("-" * 40)

            avg_token_reduction = sum(r['token_reduction_pct']
                                      for r in method_results) / len(method_results)
            avg_compression_time = sum(r['compression_time_sec']
                                       for r in method_results) / len(method_results)
            total_input_tokens = sum(r['input_tokens_used']
                                     for r in method_results)
            total_output_tokens = sum(r['output_tokens_used']
                                      for r in method_results)

            print(f"  Average token reduction: {avg_token_reduction:.1f}%")
            print(
                f"  Average compression time: {avg_compression_time:.4f} sec")
            print(f"  Total input tokens used: {total_input_tokens}")
            print(f"  Total output tokens used: {total_output_tokens}")
            print(
                f"  Total tokens used: {total_input_tokens + total_output_tokens}")

    def _save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to JSON file."""
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "client": self.client.__class__.__name__,
            "results": results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    """Example usage of the experiment runner."""

    # Test prompts
    test_prompts = [
        {
            "task_description": "Question Answering",
            "prompt": "What is the capital of France? Please provide a detailed answer explaining the history and significance of this city."
        },
        {
            "task_description": "Code Generation",
            "prompt": "Write a Python function that takes a list of numbers and returns the sum of all the even numbers in the list. Make sure to include proper error handling and documentation."
        },
        {
            "task_description": "Text Summarization",
            "prompt": "Summarize the following text: The quick brown fox jumps over the lazy dog. This is a common pangram used to test typewriters and computer keyboards because it contains every letter of the alphabet. It has been used since at least the late 1800s and remains popular today."
        },
    ]

    # Create client (defaults to OpenAI)
    print("Creating LLM client...")
    try:
        client = create_client(provider="openai", model="gpt-4o")
    except Exception as e:
        print(f"Failed to create OpenAI client: {e}")
        print("Trying Anthropic client...")
        client = create_client(provider="anthropic")

    # Run experiment
    experiment = CompressionExperiment(client)
    results = experiment.run_comparison(
        prompts=test_prompts,
        compression_methods=['none', 'caveman',
                             'llmlingua'],  # Start without llmlingua
        output_file='experiment_results.json'
    )


if __name__ == "__main__":
    main()
