"""
A tool to compress and summarize old memories to save context.

This tool analyzes old memories, identifies key points, and generates concise summaries using natural language processing techniques to extract main ideas and remove redundant details.
"""

import json
import spacy
from collections import Counter
from typing import List, Dict

nlp = spacy.load("en_core_web_sm")

def summarize_text(text: str, num_sentences: int = 3) -> str:
    """
    Summarize a given text by extracting the most important sentences.

    Args:
        text (str): The text to be summarized.
        num_sentences (int): The number of sentences to include in the summary.

    Returns:
        str: The summarized text.
    """
    doc = nlp(text)
    sentences = [sent.text for sent in list(doc.sents)]
    word_count = Counter(word.text.lower() for word in nlp(" ".join(sentences)) if word.is_alpha)
    important_sentences = sorted(sentences, key=lambda s: word_count, reverse=True)[:num_sentences]
    return " ".join(important_sentences)

def summarize_memory(memory: Dict[str, str]) -> str:
    """
    Summarize a memory by extracting the most important sentences.

    Args:
        memory (Dict[str, str]): A dictionary containing the memory details.

    Returns:
        str: The summarized memory.
    """
    return summarize_text(memory["description"])

def compress_memory(memory: Dict[str, str]) -> Dict[str, str]:
    """
    Compress a memory by removing redundant details and generating a summary.

    Args:
        memory (Dict[str, str]): A dictionary containing the memory details.

    Returns:
        Dict[str, str]: The compressed memory with a generated summary.
    """
    compressed_memory = {
        "id": memory["id"],
        "summary": summarize_memory(memory),
        "timestamp": memory["timestamp"]
    }
    return compressed_memory

def main():
    """
    Main function to demonstrate the memory summarization tool.
    """
    memory = {
        "id": "memory123",
        "description": "I went to the park today. The weather was nice and sunny. I walked around, enjoyed the view, and had a picnic with friends. We played frisbee and laughed a lot. It was a great day.",
        "timestamp": "2023-04-01"
    }
    compressed_memory = compress_memory(memory)
    print(json.dumps(compressed_memory, indent=2))

if __name__ == "__main__":
    main()