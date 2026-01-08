"""
MoverScore - Semantic similarity scorer using Word Mover Distance
Simple implementation for prompt compression evaluation.

MoverScore improves upon BERTScore by:
- Using Word Mover Distance on contextual embeddings
- Better at handling sentence reordering
- Looks at the whole "flow" of the sentence vs individual tokens

Score interpretation:
    >= 0.6: Excellent semantic preservation
    >= 0.5: Good (acceptable)
    >= 0.4: Fair (some semantic drift)
    <  0.4: Poor (significant meaning change)

Installation:
    pip install moverscore
"""

from moverscore_v2 import get_idf_dict, word_mover_score
from typing import Dict
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


def calculate_moverscore(reference: str, hypothesis: str) -> Dict:
    """
    Calculate MoverScore between two texts with detailed metrics.

    Args:
        reference: The control/reference text
        hypothesis: The compressed/test text

    Returns:
        Dictionary containing:
        - score: MoverScore (0-1, higher is better)
        - reference_length: Number of words in reference
        - hypothesis_length: Number of words in hypothesis
        - length_ratio: hypothesis_length / reference_length
        - token_reduction: Percentage of tokens reduced

    Example:
        >>> result = calculate_moverscore(
        ...     "The API returns JSON data",
        ...     "API provides JSON"
        ... )
        >>> print(f"Score: {result['score']:.4f}")
        >>> print(f"Token reduction: {result['token_reduction']:.1%}")
    """
    # Prepare inputs as lists (MoverScore expects lists)
    references = [reference]
    hypotheses = [hypothesis]

    # Get IDF dictionaries
    idf_dict_ref = get_idf_dict(references)
    idf_dict_hyp = get_idf_dict(hypotheses)

    # Calculate MoverScore
    scores = word_mover_score(
        references,
        hypotheses,
        idf_dict_ref,
        idf_dict_hyp,
        stop_words=[],  # Don't remove stop words
        n_gram=1,       # Unigram matching
        remove_subwords=True
    )

    # Extract score (returns a list, we want the first element)
    score = scores[0] if scores else 0.0

    # Calculate length metrics (split by whitespace)
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
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
    print("MoverScore - Example Usage")
    print("=" * 70)
    print("Note: First run downloads pretrained models (~500MB)\n")

    # Example 1: Basic comparison
    print("Example 1: Basic Text Comparison")
    print("-" * 70)

    reference = "The quick brown fox jumps over the lazy dog"
    hypothesis = "A fast brown fox leaps over the sleepy dog"

    result = calculate_moverscore(reference, hypothesis)

    print(f"Reference:  {reference}")
    print(f"Hypothesis: {hypothesis}")
    print(f"\nMoverScore: {result['score']:.4f}")
    print(f"Tokens: {result['reference_length']} → {result['hypothesis_length']}")

    # Example 2: Prompt compression evaluation
    print("\n\nExample 2: Prompt Compression Evaluation")
    print("-" * 70)

    control = "The API returns a JSON object containing user data, including name, email, and registration date"
    compressed = "API provides JSON with user information: name, email, signup date"

    result = calculate_moverscore(control, compressed)

    print(f"Control Output:")
    print(f"  {control}")
    print(f"\nCompressed Output:")
    print(f"  {compressed}")
    print(f"\nResults:")
    print(f"  MoverScore: {result['score']:.4f}")
    print(f"  Control tokens: {result['reference_length']}")
    print(f"  Compressed tokens: {result['hypothesis_length']}")
    print(f"  Length ratio: {result['length_ratio']:.2%}")
    print(f"  Token reduction: {result['token_reduction']:.2%}")
