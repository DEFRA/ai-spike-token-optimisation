"""
Oreo Technique Experiment - Compress only the data, not the instructions.

This experiment tests the "Oreo cookie" approach:
- Top Cookie (System Instructions): Keep RAW
- Cream (The Content/Data): COMPRESS THIS
- Bottom Cookie (Query): Keep RAW

This should maximize token savings on the bulk content while maintaining
clear instructions and queries for the LLM.
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from llm_client import create_client, LLMClient
from compressors import create_compressor, Compressor


@dataclass
class OreoPrompt:
    """
    Represents a prompt in Oreo format.

    Attributes:
        system_instructions: Top cookie - clear instructions for the model
        content: Cream - the bulk data/context to be compressed
        query: Bottom cookie - the specific question/task
        task_description: Human-readable description of the task
    """
    system_instructions: str
    content: str
    query: str
    task_description: str


class OreoExperiment:
    """Run experiments using the Oreo compression technique."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize Oreo experiment runner.

        Args:
            llm_client: LLM client to use for all experiments
        """
        self.client = llm_client

    def build_full_prompt(
        self,
        oreo_prompt: OreoPrompt,
        compress_content: bool = False,
        compressor: Compressor = None
    ) -> str:
        """
        Build a complete prompt from Oreo components.

        Args:
            oreo_prompt: The Oreo-structured prompt
            compress_content: Whether to compress the middle content
            compressor: Compressor to use (required if compress_content=True)

        Returns:
            Complete prompt string
        """
        # Get the content (compressed or not)
        if compress_content and compressor is not None:
            content = compressor.compress(oreo_prompt.content)
        else:
            content = oreo_prompt.content

        # Assemble the full prompt
        parts = []

        if oreo_prompt.system_instructions:
            parts.append(oreo_prompt.system_instructions)

        if content:
            parts.append(content)

        if oreo_prompt.query:
            parts.append(oreo_prompt.query)

        return "\n\n".join(parts)

    def run_single_test(
        self,
        oreo_prompt: OreoPrompt,
        compressor: Compressor,
        method_name: str,
        compress_content: bool = True
    ) -> Dict[str, Any]:
        """
        Run a single test with the Oreo technique.

        Args:
            oreo_prompt: The Oreo-structured prompt
            compressor: Compressor to use
            method_name: Name of the method for logging
            compress_content: Whether to compress the content section

        Returns:
            Dictionary with test results
        """
        # Build the full prompt (with or without compression)
        start_time = time.time()
        full_prompt = self.build_full_prompt(
            oreo_prompt,
            compress_content=compress_content,
            compressor=compressor
        )
        compression_time = time.time() - start_time

        # Build uncompressed version for comparison
        uncompressed_prompt = self.build_full_prompt(oreo_prompt, compress_content=False)

        # Count tokens for each component
        instructions_tokens = self.client.count_tokens(oreo_prompt.system_instructions) if oreo_prompt.system_instructions else 0
        query_tokens = self.client.count_tokens(oreo_prompt.query) if oreo_prompt.query else 0

        # Content tokens (original and compressed)
        original_content_tokens = self.client.count_tokens(oreo_prompt.content)
        if compress_content:
            compressed_content = compressor.compress(oreo_prompt.content)
            compressed_content_tokens = self.client.count_tokens(compressed_content)
        else:
            compressed_content = oreo_prompt.content
            compressed_content_tokens = original_content_tokens

        # Total tokens
        total_original_tokens = instructions_tokens + original_content_tokens + query_tokens
        total_compressed_tokens = instructions_tokens + compressed_content_tokens + query_tokens

        # Send to LLM
        response, token_stats = self.client.send_prompt(full_prompt)

        # Calculate metrics
        content_compression_ratio = len(compressed_content) / len(oreo_prompt.content) if len(oreo_prompt.content) > 0 else 1.0
        content_token_reduction = original_content_tokens - compressed_content_tokens
        content_token_reduction_pct = (content_token_reduction / original_content_tokens * 100) if original_content_tokens > 0 else 0

        total_token_reduction = total_original_tokens - total_compressed_tokens
        total_token_reduction_pct = (total_token_reduction / total_original_tokens * 100) if total_original_tokens > 0 else 0

        return {
            "method": method_name,
            "task_description": oreo_prompt.task_description,
            "compress_content": compress_content,

            # Component details
            "instructions_tokens": instructions_tokens,
            "query_tokens": query_tokens,
            "original_content_tokens": original_content_tokens,
            "compressed_content_tokens": compressed_content_tokens,
            "content_token_reduction": content_token_reduction,
            "content_token_reduction_pct": content_token_reduction_pct,
            "content_compression_ratio": content_compression_ratio,

            # Total metrics
            "total_original_tokens": total_original_tokens,
            "total_compressed_tokens": total_compressed_tokens,
            "total_token_reduction": total_token_reduction,
            "total_token_reduction_pct": total_token_reduction_pct,

            # LLM usage
            "input_tokens_used": token_stats["input_tokens"],
            "output_tokens_used": token_stats["output_tokens"],
            "total_tokens_used": token_stats["total_tokens"],

            # Timing
            "compression_time_sec": compression_time,

            # Full texts for analysis
            "original_prompt": uncompressed_prompt,
            "compressed_prompt": full_prompt,
            "response": response,
            "original_content": oreo_prompt.content,
            "compressed_content": compressed_content if compress_content else oreo_prompt.content,
        }

    def run_comparison(
        self,
        oreo_prompts: List[OreoPrompt],
        compression_methods: List[str] = None,
        output_file: str = None
    ) -> List[Dict[str, Any]]:
        """
        Run Oreo comparison across multiple prompts and compression methods.

        For each prompt, tests:
        1. Control: No compression (baseline)
        2. Test: Compress ONLY the content (Oreo technique)

        Args:
            oreo_prompts: List of OreoPrompt objects
            compression_methods: List of method names to test (e.g., ['caveman', 'llmlingua'])
            output_file: Optional file to save results as JSON

        Returns:
            List of all test results
        """
        if compression_methods is None:
            compression_methods = ['caveman', 'llmlingua']

        all_results = []

        print(f"\n{'='*80}")
        print(f"Starting Oreo Technique Experiment")
        print(f"{'='*80}")
        print(f"Number of test prompts: {len(oreo_prompts)}")
        print(f"Compression methods: {', '.join(compression_methods)}")
        print(f"LLM Client: {self.client.__class__.__name__}")
        print(f"{'='*80}\n")

        for idx, oreo_prompt in enumerate(oreo_prompts):
            print(f"\n--- Test {idx + 1}/{len(oreo_prompts)}: {oreo_prompt.task_description} ---")

            # First, run control (no compression)
            print(f"\nControl: No compression")
            try:
                control_result = self.run_single_test(
                    oreo_prompt=oreo_prompt,
                    compressor=create_compressor('none'),
                    method_name='none',
                    compress_content=False
                )
                all_results.append(control_result)

                print(f"  Total tokens: {control_result['total_original_tokens']}")
                print(f"  Input tokens used: {control_result['input_tokens_used']}")
                print(f"  Output tokens used: {control_result['output_tokens_used']}")
                print(f"  Response length: {len(control_result['response'])} chars")
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                continue

            # Test each compression method (Oreo technique)
            for method in compression_methods:
                print(f"\nOreo Technique: {method} (content only)")

                # Create compressor
                if method == 'llmlingua':
                    compressor = create_compressor(method, rate=0.5)
                else:
                    compressor = create_compressor(method)

                # Run test
                try:
                    result = self.run_single_test(
                        oreo_prompt=oreo_prompt,
                        compressor=compressor,
                        method_name=f'{method}_oreo',
                        compress_content=True
                    )
                    all_results.append(result)

                    # Print summary
                    print(f"  Content tokens: {result['original_content_tokens']} → {result['compressed_content_tokens']}")
                    print(f"  Content reduction: {result['content_token_reduction_pct']:.1f}%")
                    print(f"  Total tokens: {result['total_original_tokens']} → {result['total_compressed_tokens']}")
                    print(f"  Total reduction: {result['total_token_reduction_pct']:.1f}%")
                    print(f"  Input tokens used: {result['input_tokens_used']}")
                    print(f"  Output tokens used: {result['output_tokens_used']}")
                    print(f"  Response length: {len(result['response'])} chars")

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
        """Print summary statistics comparing control vs Oreo technique."""
        print(f"\n{'='*80}")
        print("OREO EXPERIMENT SUMMARY")
        print(f"{'='*80}\n")

        # Group by method
        by_method = {}
        for result in results:
            method = result['method']
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)

        # Calculate averages for each method
        for method, method_results in by_method.items():
            print(f"\n{method.upper()} METHOD:")
            print("-" * 40)

            # Averages
            avg_total_reduction = sum(r['total_token_reduction_pct'] for r in method_results) / len(method_results)
            avg_compression_time = sum(r['compression_time_sec'] for r in method_results) / len(method_results)
            total_input_tokens = sum(r['input_tokens_used'] for r in method_results)
            total_output_tokens = sum(r['output_tokens_used'] for r in method_results)

            print(f"  Average total token reduction: {avg_total_reduction:.1f}%")
            print(f"  Average compression time: {avg_compression_time:.4f} sec")
            print(f"  Total input tokens used: {total_input_tokens}")
            print(f"  Total output tokens used: {total_output_tokens}")
            print(f"  Total tokens used: {total_input_tokens + total_output_tokens}")

            # If this is an Oreo method, show content-specific stats
            if '_oreo' in method:
                avg_content_reduction = sum(r['content_token_reduction_pct'] for r in method_results) / len(method_results)
                avg_instructions_tokens = sum(r['instructions_tokens'] for r in method_results) / len(method_results)
                avg_query_tokens = sum(r['query_tokens'] for r in method_results) / len(method_results)

                print(f"\n  Oreo Breakdown:")
                print(f"    Avg content reduction: {avg_content_reduction:.1f}%")
                print(f"    Avg instruction tokens (preserved): {avg_instructions_tokens:.1f}")
                print(f"    Avg query tokens (preserved): {avg_query_tokens:.1f}")

    def _save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to JSON file."""
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "experiment_type": "oreo_technique",
            "client": self.client.__class__.__name__,
            "results": results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    """Example usage of the Oreo experiment runner."""

    # Create test prompts in Oreo format
    oreo_prompts = [
        OreoPrompt(
            task_description="Article Analysis",
            system_instructions="You are a helpful assistant. Be concise and answer in clear, complete sentences.",
            content="""
            Climate change is one of the most pressing issues facing humanity today. The scientific consensus is clear:
            global temperatures are rising at an unprecedented rate, primarily due to human activities such as burning
            fossil fuels, deforestation, and industrial processes. The Intergovernmental Panel on Climate Change (IPCC)
            has documented extensive evidence showing that the average global temperature has increased by approximately
            1.1°C since the pre-industrial era. This warming trend has led to numerous observable effects, including
            melting polar ice caps, rising sea levels, more frequent and severe weather events, and shifts in wildlife
            populations and habitats.

            The impacts of climate change are far-reaching and affect every aspect of human society and natural ecosystems.
            Coastal communities face increased flooding risks, agricultural systems are disrupted by changing precipitation
            patterns, and extreme weather events cause billions of dollars in damage annually. Scientists warn that without
            significant action to reduce greenhouse gas emissions, these effects will intensify, potentially leading to
            catastrophic consequences for future generations.

            Addressing climate change requires a multifaceted approach involving governments, businesses, and individuals.
            Transitioning to renewable energy sources, improving energy efficiency, protecting and restoring forests, and
            developing sustainable agricultural practices are all crucial steps. International cooperation, as exemplified
            by agreements like the Paris Climate Accord, plays a vital role in coordinating global efforts to limit
            temperature increases and adapt to unavoidable changes.
            """,
            query="What is the main conclusion of this article?"
        ),

        OreoPrompt(
            task_description="Code Documentation Analysis",
            system_instructions="You are a code documentation expert. Answer in JSON format with keys: 'purpose', 'main_functions', 'complexity'.",
            content="""
            def calculate_fibonacci(n, memo={}):
                '''
                Calculate the nth Fibonacci number using memoization for efficiency.

                The Fibonacci sequence is a series of numbers where each number is the sum of the
                two preceding ones, usually starting with 0 and 1. This implementation uses dynamic
                programming with memoization to avoid redundant calculations and improve performance
                from exponential to linear time complexity.

                Args:
                    n (int): The position in the Fibonacci sequence (0-indexed)
                    memo (dict): Dictionary to store previously calculated values

                Returns:
                    int: The nth Fibonacci number

                Examples:
                    >>> calculate_fibonacci(0)
                    0
                    >>> calculate_fibonacci(1)
                    1
                    >>> calculate_fibonacci(10)
                    55

                Time Complexity: O(n)
                Space Complexity: O(n)
                '''
                if n in memo:
                    return memo[n]
                if n <= 1:
                    return n

                memo[n] = calculate_fibonacci(n-1, memo) + calculate_fibonacci(n-2, memo)
                return memo[n]

            def fibonacci_sequence(count):
                '''
                Generate a list of the first 'count' Fibonacci numbers.

                This function creates a list containing the first 'count' numbers in the
                Fibonacci sequence by repeatedly calling the calculate_fibonacci function.
                '''
                return [calculate_fibonacci(i) for i in range(count)]
            """,
            query="Summarize this code's purpose and main functions."
        ),

        OreoPrompt(
            task_description="Product Review Summary",
            system_instructions="You are a review analyzer. Extract the key sentiment and main points. Be concise.",
            content="""
            I recently purchased the UltraWidget Pro 3000 and I have to say, I'm thoroughly impressed with this product.
            Right out of the box, the build quality is exceptional - it feels solid and well-constructed, with attention
            to detail in every aspect. The sleek design fits perfectly in my workspace and the matte black finish looks
            professional and modern.

            In terms of performance, the UltraWidget Pro 3000 exceeds expectations. It handles everything I throw at it
            with ease, from basic tasks to more demanding operations. The battery life is outstanding - I can go a full
            day of heavy use without needing to recharge. The display is crisp and vibrant, making it a pleasure to use
            for extended periods.

            The user interface is intuitive and easy to navigate, even for someone who isn't particularly tech-savvy.
            Setup was a breeze - I had it up and running in less than 10 minutes. The accompanying mobile app works
            flawlessly and adds convenient remote control functionality.

            Customer support has been excellent. When I had a minor question about one of the features, I reached their
            support team via chat and received helpful, detailed assistance within minutes. They clearly stand behind
            their product.

            The only minor drawback I've noticed is that it can get slightly warm during intensive use, but this hasn't
            affected performance at all and is barely noticeable. For the price point, this is an exceptional value.
            I would highly recommend the UltraWidget Pro 3000 to anyone looking for a reliable, high-quality solution.
            """,
            query="What is the overall sentiment and key points of this review?"
        ),
    ]

    # Use environment variable for provider selection
    print("Creating LLM client...")
    client = create_client()

    # Run Oreo experiment
    experiment = OreoExperiment(client)
    results = experiment.run_comparison(
        oreo_prompts=oreo_prompts,
        compression_methods=['caveman', 'llmlingua'],
        output_file='oreo_experiment_results.json'
    )

    print(f"\n{'='*80}")
    print("Experiment complete! Check oreo_experiment_results.json for full details.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
