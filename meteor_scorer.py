"""
METEOR (Metric for Evaluation of Translation with Explicit ORdering) Scorer
Simple implementation for prompt compression evaluation.

METEOR improves upon ROUGE by considering:
- Exact word matches
- Stemming (e.g., "running" matches "run")
- Synonyms (using WordNet)

Score interpretation:
    >= 0.8: Excellent quality preservation
    >= 0.6: Good (acceptable)
    >= 0.4: Fair (some quality loss)
    <  0.4: Poor (significant degradation)
"""

import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import os
from typing import Dict


def setup_nltk_resources():
    """Download required NLTK resources (run once)."""
    resources = ['wordnet', 'punkt_tab', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}/')
        except LookupError:
            try:
                nltk.data.find(f'tokenizers/{resource}/')
            except LookupError:
                print('setting up nltk resources')
                nltk.download(resource, quiet=True)


def calculate_meteor(reference: str, hypothesis: str) -> Dict:
    """
    Calculate METEOR score between two texts with detailed metrics.

    Args:
        reference: The control/reference text
        hypothesis: The compressed/test text

    Returns:
        Dictionary containing:
        - score: METEOR score (0-1, higher is better)
        - reference_length: Number of tokens in reference
        - hypothesis_length: Number of tokens in hypothesis
        - length_ratio: hypothesis_length / reference_length
        - token_reduction: Percentage of tokens reduced

    Example:
        >>> result = calculate_meteor(
        ...     "The API returns JSON data",
        ...     "API provides JSON"
        ... )
        >>> print(f"Score: {result['score']:.4f}")
        >>> print(f"Token reduction: {result['token_reduction']:.1%}")
    """
    # Ensure resources are available
    setup_nltk_resources()

    # Tokenize
    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())

    # Calculate score
    score = meteor_score([ref_tokens], hyp_tokens)

    # Calculate metrics
    length_ratio = len(hyp_tokens) / len(ref_tokens) if ref_tokens else 0.0
    token_reduction = 1 - length_ratio

    return {
        'score': score,
        'reference_length': len(ref_tokens),
        'hypothesis_length': len(hyp_tokens),
        'length_ratio': length_ratio,
        'token_reduction': token_reduction
    }


if __name__ == "__main__":
    # Example usage
    print("METEOR Scorer - Example Usage")
    print("=" * 70)

    # Example 1: Basic comparison
    print("\nExample 1: Basic Text Comparison")
    print("-" * 70)

    reference = "The quick brown fox jumps over the lazy dog"
    hypothesis = "A fast brown fox leaps over the sleepy dog"

    result = calculate_meteor(reference, hypothesis)

    print(f"Reference:  {reference}")
    print(f"Hypothesis: {hypothesis}")
    print(f"\nMETEOR Score: {result['score']:.4f}")
    print(f"Tokens: {result['reference_length']} → {result['hypothesis_length']}")

    # Example 2: Prompt compression evaluation
    print("\n\nExample 2: Prompt Compression Evaluation")
    print("-" * 70)

    control = "The API returns a JSON object containing user data, including name, email, and registration date"
    compressed = "API provides JSON with user information: name, email, signup date"

    result = calculate_meteor(control, compressed)

    print(f"Control Output:")
    print(f"  {control}")
    print(f"\nCompressed Output:")
    print(f"  {compressed}")
    print(f"\nResults:")
    print(f"  METEOR Score: {result['score']:.4f}")
    print(f"  Control tokens: {result['reference_length']}")
    print(f"  Compressed tokens: {result['hypothesis_length']}")
    print(f"  Length ratio: {result['length_ratio']:.2%}")
    print(f"  Token reduction: {result['token_reduction']:.2%}")