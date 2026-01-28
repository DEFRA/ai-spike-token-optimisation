"""
Example usage demonstrating individual components.
"""

from compressors import create_compressor
from llm_client import create_client
import os

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def demo_compression_methods():
    """Demonstrate different compression methods."""
    print("="*80)
    print("COMPRESSION METHODS DEMO")
    print("="*80)

    test_text = """
    The quick brown fox jumps over the lazy dog. This is a very common pangram
    that has been used for testing typewriters and computer keyboards because it
    contains every single letter of the alphabet. It has been in use since at
    least the late 1800s and it remains popular today in the modern era.
    """

    print(f"\nOriginal text ({len(test_text)} chars):")
    print(test_text)

    # No compression
    print("\n" + "-"*80)
    print("NO COMPRESSION (Baseline):")
    print("-"*80)
    no_comp = create_compressor('none')
    result = no_comp.compress(test_text)
    print(f"Result ({len(result)} chars):")
    print(result)

    # Caveman compression
    print("\n" + "-"*80)
    print("CAVEMAN COMPRESSION:")
    print("-"*80)
    caveman = create_compressor('caveman')
    result = caveman.compress(test_text)
    print(f"Result ({len(result)} chars):")
    print(result)
    print(
        f"Compression ratio: {caveman.get_compression_ratio(test_text, result):.2%}")

    # LLMLingua compression (if available)
    print("\n" + "-"*80)
    print("LLMLINGUA-2 COMPRESSION:")
    print("-"*80)
    try:
        llmlingua = create_compressor('llmlingua', rate=0.5)
        result = llmlingua.compress(test_text)
        print(f"Result ({len(result)} chars):")
        print(result)
        print(
            f"Compression ratio: {llmlingua.get_compression_ratio(test_text, result):.2%}")
    except ImportError as e:
        print(f"LLMLingua not available: {e}")


def demo_llm_client():
    """Demonstrate LLM client with token counting."""
    print("\n" + "="*80)
    print("LLM CLIENT DEMO")
    print("="*80)

    prompt = "What is 2+2? Answer briefly."

    try:
        # Use environment variable for provider selection
        print("\nCreating LLM client from environment variables...")
        client = create_client(provider=DEFAULT_PROVIDER)

        print(f"\nPrompt: {prompt}")
        print(f"Token count: {client.count_tokens(prompt)}")

        response, stats = client.send_prompt(prompt)

        print(f"\nResponse: {response}")
        print(f"\nToken statistics:")
        print(f"  Input tokens: {stats['input_tokens']}")
        print(f"  Output tokens: {stats['output_tokens']}")
        print(f"  Total tokens: {stats['total_tokens']}")

    except Exception as e:
        print(f"LLM client failed: {e}")
        print("\nPlease set DEFAULT_PROVIDER, ANTHROPIC_MODEL (if using Anthropic), and appropriate API key in .env file.")
        print("Example .env settings:")
        print("  DEFAULT_PROVIDER=anthropic")
        print("  ANTHROPIC_MODEL=claude-sonnet-4-20250514")
        print("  ANTHROPIC_API_KEY=your-key-here")


def demo_end_to_end():
    """Demonstrate end-to-end compression + LLM."""
    print("\n" + "="*80)
    print("END-TO-END DEMO: COMPRESSION + LLM")
    print("="*80)

    prompt = """
    Please write a short Python function that calculates the factorial of a number.
    The function should include proper error handling for negative numbers and should
    be well documented with docstrings. Make sure to include type hints as well.
    """

    print(f"\nOriginal prompt ({len(prompt)} chars):")
    print(prompt)

    # Compress with caveman
    print("\n" + "-"*80)
    print("CAVEMAN COMPRESSION")
    print("-"*80)
    caveman = create_compressor('caveman')
    caveman_compressed = caveman.compress(prompt)

    print(f"Caveman compressed ({len(caveman_compressed)} chars):")
    print(caveman_compressed)
    print(
        f"Compression ratio: {caveman.get_compression_ratio(prompt, caveman_compressed):.2%}")

    # Compress with LLMLingua-2
    print("\n" + "-"*80)
    print("LLMLINGUA-2 COMPRESSION")
    print("-"*80)
    try:
        llmlingua = create_compressor('llmlingua', rate=0.5)
        llmlingua_compressed = llmlingua.compress(prompt)

        print(f"LLMLingua-2 compressed ({len(llmlingua_compressed)} chars):")
        print(llmlingua_compressed)
        print(
            f"Compression ratio: {llmlingua.get_compression_ratio(prompt, llmlingua_compressed):.2%}")
    except Exception as e:
        print(f"LLMLingua-2 not available: {e}")
        llmlingua_compressed = None

    # Send to LLM
    print("\n" + "-"*80)
    print("SENDING TO LLM")
    print("-"*80)
    try:
        # Respect DEFAULT_PROVIDER env var
        client = create_client(provider=DEFAULT_PROVIDER)

        print("\n=== Testing Basic LLM Client ===")
        response1, stats1 = client.send_prompt(prompt)
        print(f"Input tokens: {stats1['input_tokens']}")
        print(f"Output: {response1}...")

        print("\n--- Caveman compressed prompt ---")
        response2, stats2 = client.send_prompt(caveman_compressed)
        print(f"Input tokens: {stats2['input_tokens']}")
        print(
            f"Token savings: {stats1['input_tokens'] - stats2['input_tokens']} ({(stats1['input_tokens'] - stats2['input_tokens']) / stats1['input_tokens'] * 100:.1f}%)")
        print(f"Output: {response2}...")

        if llmlingua_compressed:
            print("\n--- LLMLingua-2 compressed prompt ---")
            response3, stats3 = client.send_prompt(llmlingua_compressed)
            print(f"Input tokens: {stats3['input_tokens']}")
            print(
                f"Token savings: {stats1['input_tokens'] - stats3['input_tokens']} ({(stats1['input_tokens'] - stats3['input_tokens']) / stats1['input_tokens'] * 100:.1f}%)")
            print(f"Output: {response3}...")

    except Exception as e:
        print(f"LLM client not available: {e}")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test with actual LLM.")


if __name__ == "__main__":
    demo_compression_methods()
    demo_llm_client()
    demo_end_to_end()
