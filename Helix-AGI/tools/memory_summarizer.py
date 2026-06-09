"""
A tool to compress old memories by identifying redundant or less important details and generating a summary that captures the key points while removing unnecessary context.
"""

import json
from typing import List, Dict
from collections import Counter
from functools import reduce

class MemorySummarizer:
    def __init__(self, memories: List[Dict]):
        self.memories = memories

    def summarize(self, threshold: float = 0.8) -> Dict:
        """
        Summarize the memories by identifying redundant or less important details and generating a compressed summary.

        Args:
            threshold (float): The minimum frequency threshold for considering a word as important. Defaults to 0.8.

        Returns:
            Dict: A compressed summary of the memories.
        """
        # Flatten the memories into a single list of words
        words = reduce(lambda x, y: x + y, [memory['content'].split() for memory in self.memories])

        # Count the frequency of each word
        word_counts = Counter(words)

        # Identify important words based on the frequency threshold
        important_words = {word for word, count in word_counts.items() if count / len(words) >= threshold}

        # Generate the summary by reconstructing the memories using only the important words
        summary = {
            'summary': ' '.join(important_words),
            'important_words': list(important_words)
        }

        return summary

def main():
    # Example usage
    memories = [
        {'content': 'The cat sat on the mat. The dog barked at the mailman.'},
        {'content': 'The cat chased the ball. The dog barked at the mailman.'},
        {'content': 'The cat and the dog played in the garden.'}
    ]

    summarizer = MemorySummarizer(memories)
    summary = summarizer.summarize(threshold=0.6)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()