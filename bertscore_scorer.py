"""
BERTScore - Semantic similarity scorer using contextual embeddings
Simple implementation for prompt compression evaluation.

BERTScore improves upon lexical metrics by:
- Using pre-trained transformer embeddings (like RoBERTa)
- Calculating cosine similarity between contextual embeddings
- Measuring semantic similarity rather than exact word matches

Score interpretation:
    >= 0.9: Excellent semantic preservation
    >= 0.8: Good (acceptable)
    >= 0.7: Fair (some semantic drift)
    <  0.7: Poor (significant meaning change)

Installation:
    pip install bert-score


Precision - "How much of the hypothesis is relevant?"
Measures what % of words in the compressed output have matching counterparts in the control output
High precision: Compressed output doesn't add irrelevant/hallucinated information
Low precision: Compressed output added words not semantically related to control
Example:
Control: "The cat sat on the mat"
Compressed: "The cat sat on the mat and ate fish"
Low precision: Added "and ate fish" which wasn't in the control


Recall - "How much of the reference is covered?"
Measures what % of words in the control output are represented in the compressed output
High recall: Compressed output captures most of the control's meaning
Low recall: Compressed output missed important information from control
Example:
Control: "The cat sat on the mat"
Compressed: "Cat sat"
Low recall: Missing "the", "on", "the mat" - loses detail


F1 Score - "Overall similarity balance"
Harmonic mean of precision and recall
This is the main score to use for compression evaluation
Balances both "did it add nonsense?" and "did it lose meaning?"


For Prompt Compression Evaluation:
Metric	What You Want	What It Tells You
F1	≥ 0.85	Overall semantic similarity (main metric)
Precision	≥ 0.80	Compressed output didn't hallucinate
Recall	≥ 0.80	Compressed output captured the meaning

"""

from bert_score import score as bert_score
from typing import Dict


def calculate_bertscore(reference: str, hypothesis: str, model: str = "distilbert-base-uncased") -> Dict:
    """
    Calculate BERTScore between two texts with detailed metrics.

    Args:
        reference: The control/reference text
        hypothesis: The compressed/test text
        model: Pre-trained model to use (default: distilbert-base-uncased)
               Other options: "roberta-large", "bert-base-uncased"

    Returns:
        Dictionary containing:
        - precision: BERTScore precision (0-1)
        - recall: BERTScore recall (0-1)
        - f1: BERTScore F1 score (0-1, higher is better)
        - reference_length: Number of words in reference
        - hypothesis_length: Number of words in hypothesis
        - length_ratio: hypothesis_length / reference_length
        - token_reduction: Percentage of tokens reduced

    Example:
        >>> result = calculate_bertscore(
        ...     "The API returns JSON data",
        ...     "API provides JSON"
        ... )
        >>> print(f"F1 Score: {result['f1']:.4f}")
        >>> print(f"Token reduction: {result['token_reduction']:.1%}")
    """
    # Calculate BERTScore (returns tensors)
    P, R, F1 = bert_score(
        [hypothesis],
        [reference],
        model_type=model,
        verbose=False
    )

    # Convert to Python floats
    precision = P.item()
    recall = R.item()
    f1_score = F1.item()

    return {
        # 'precision': precision,
        # 'recall': recall,
        'method': 'bert',
        'score': f1_score,
    }


if __name__ == "__main__":
    # Example usage
    print("BERTScore - Example Usage")
    print("=" * 70)
    print("Note: First run downloads pretrained models\n")

    # Example 1: Basic comparison
    print("Example 1: Basic Text Comparison")
    print("-" * 70)

    reference = "The quick brown fox jumps over the lazy dog"
    hypothesis = "A fast brown fox leaps over the sleepy dog"

    print("Calculating BERTScore...")
    result = calculate_bertscore(reference, hypothesis)

    print(f"Reference:  {reference}")
    print(f"Hypothesis: {hypothesis}")
    print(f"\nBERTScore F1: {result['f1']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(
        f"Tokens: {result['reference_length']} → {result['hypothesis_length']}")

    # Example 2: Prompt compression evaluation
    print("\n\nExample 2: Prompt Compression Evaluation")
    print("-" * 70)

    control = "The API returns a JSON object containing user data, including name, email, and registration date"
    compressed = "API provides JSON with user information: name, email, signup date"

    print("Calculating BERTScore...")
    result = calculate_bertscore(control, compressed)

    print(f"Control Output:")
    print(f"  {control}")
    print(f"\nCompressed Output:")
    print(f"  {compressed}")
    print(f"\nResults:")
    print(f"  BERTScore F1: {result['f1']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  Control tokens: {result['reference_length']}")
    print(f"  Compressed tokens: {result['hypothesis_length']}")
    print(f"  Length ratio: {result['length_ratio']:.2%}")
    print(f"  Token reduction: {result['token_reduction']:.2%}")
