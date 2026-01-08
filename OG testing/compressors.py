"""
Prompt compression methods for LLM inputs.
"""

import os
import re
from typing import Optional
from abc import ABC, abstractmethod

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Compressor(ABC):
    """Base class for prompt compressors."""

    @abstractmethod
    def compress(self, text: str) -> str:
        """Compress the input text."""
        pass

    def get_compression_ratio(self, original: str, compressed: str) -> float:
        """Calculate compression ratio (compressed/original length)."""
        return len(compressed) / len(original) if len(original) > 0 else 0.0


class NoCompression(Compressor):
    """Baseline - no compression applied."""

    def compress(self, text: str) -> str:
        """Return text unchanged."""
        return text


class CavemanCompressor(Compressor):
    """
    Simple compression that drops unimportant words to create "caveman talk".
    Removes articles, auxiliary verbs, some prepositions, etc.
    """

    def __init__(self):
        # Words to remove - articles, auxiliary verbs, some common prepositions
        self.stopwords = {
            # Articles
            'a', 'an', 'the',
            # Auxiliary verbs
            'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
            'has', 'have', 'had', 'having',
            'do', 'does', 'did', 'doing',
            'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',
            # Common but often non-essential words
            'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'from',
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once',
            # Pronouns (keeping some key ones, removing others)
            'this', 'that', 'these', 'those',
            # Other
            'there', 'here', 'where',
            'very', 'just', 'really', 'quite', 'too', 'so',
        }

    def compress(self, text: str) -> str:
        """
        Compress text by removing unimportant words.
        Preserves sentence structure but creates "caveman" style English.
        """
        # Split into sentences to preserve some structure
        sentences = re.split(r'([.!?]+)', text)

        compressed_sentences = []
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ''

            # Tokenize words (simple split, preserving contractions)
            words = sentence.split()

            # Keep words that aren't in stopwords
            filtered_words = []
            for word in words:
                # Extract the actual word (lowercase) for checking
                clean_word = re.sub(r'[^\w\']', '', word.lower())

                # Keep if not a stopword, or if it's at the start of sentence
                if clean_word not in self.stopwords or len(filtered_words) == 0:
                    filtered_words.append(word)

            if filtered_words:
                compressed_sentences.append(' '.join(filtered_words) + punctuation)

        result = ' '.join(compressed_sentences).strip()

        # Clean up extra spaces
        result = re.sub(r'\s+([.!?,;:])', r'\1', result)
        result = re.sub(r'\s+', ' ', result)

        return result


class LLMLinguaCompressor(Compressor):
    """
    Compression using LLMLingua-2.
    Uses trained models for selective prompt compression.
    """

    def __init__(self, target_token: Optional[int] = None, rate: Optional[float] = 0.5):
        """
        Initialize LLMLingua-2 compressor.

        Args:
            target_token: Target number of tokens (if specified, rate is ignored)
            rate: Compression rate (0.0 to 1.0), lower = more compression
        """
        try:
            from llmlingua import PromptCompressor
            import torch
        except ImportError:
            raise ImportError(
                "llmlingua package not installed. Run: pip install llmlingua\n"
                "Note: This also requires transformers and torch."
            )

        # Detect and use the best available device
        if torch.backends.mps.is_available():
            device = "mps"
            print("Initializing LLMLingua-2 compressor (using MPS - Apple Silicon GPU)...")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Initializing LLMLingua-2 compressor (using CUDA)...")
        else:
            device = "cpu"
            print("Initializing LLMLingua-2 compressor (using CPU)...")

        self.compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map=device,
        )
        self.target_token = target_token
        self.rate = rate

    def compress(self, text: str) -> str:
        """
        Compress text using LLMLingua-2.

        Returns:
            Compressed text
        """
        # LLMLingua-2 expects text as a list of strings or single string
        # Only pass target_token if it's set, otherwise use rate
        compress_kwargs = {
            'force_tokens': ['\n', '?', '!', '.'],  # Preserve important punctuation
            'use_sentence_level_filter': True,
            'use_context_level_filter': True,
            'use_token_level_filter': True,
        }

        if self.target_token is not None:
            compress_kwargs['target_token'] = self.target_token
        else:
            compress_kwargs['rate'] = self.rate

        result = self.compressor.compress_prompt(text, **compress_kwargs)

        return result['compressed_prompt']


def create_compressor(method: str, **kwargs) -> Compressor:
    """
    Factory function to create compressors.

    Args:
        method: 'none', 'caveman', or 'llmlingua'
        **kwargs: Additional arguments for the compressor

    Returns:
        Compressor instance
    """
    method = method.lower()

    if method == 'none':
        return NoCompression()
    elif method == 'caveman':
        return CavemanCompressor()
    elif method == 'llmlingua' or method == 'llmlingua2':
        return LLMLinguaCompressor(**kwargs)
    else:
        raise ValueError(
            f"Unknown compression method: {method}. "
            f"Use 'none', 'caveman', or 'llmlingua'."
        )
