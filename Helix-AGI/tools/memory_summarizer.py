"""
A tool to compress old memories by generating concise summaries.

This tool analyzes old memories, identifies patterns and key information, and generates concise summaries to replace the full memories. This allows the system to retain the essential context while reducing the storage footprint.
"""

import json
from typing import List, Dict
from collections import Counter
from functools import reduce
from itertools import chain

class MemorySummarizer:
    def __init__(self, memories: List[Dict]):
        self.memories = memories

    def summarize(self) -> List[Dict]:
        # Extract all words from all memories
        all_words = list(chain.from_iterable([list(m['content'].values()) for m in self.memories]))

        # Count word frequencies
        word_counts = Counter(all_words)

        # Get the most common words
        common_words = [w for w, c in word_counts.most_common(10)]

        # Replace each memory with a summary
        summaries = []
        for memory in self.memories:
            summary = {}
            for key, value in memory.items():
                if key != 'summary':
                    summary[key] = ' '.join([w for w in value.split() if w not in common_words])
            summary['summary'] = ' '.join(common_words)
            summaries.append(summary)

        return summaries

def main():
    with open('memories.json') as f:
        memories = json.load(f)['memories']

    summarizer = MemorySummarizer(memories)
    summarized_memories = summarizer.summarize()

    for memory in summarized_memories:
        print(json.dumps(memory, indent=2))

if __name__ == '__main__':
    main()