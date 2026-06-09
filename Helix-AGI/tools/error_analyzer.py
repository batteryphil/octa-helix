"""
A tool to analyze error-containing responses, extract key details, and identify common error types and contexts.

Usage:
1. Run the script with a list of error-containing responses as input.
2. The tool will parse the responses, extract key details, and generate a report.
3. Review the generated report to identify patterns and common error types.
"""

import re
import sys
from collections import defaultdict

class ErrorAnalyzer:
    def __init__(self, responses):
        self.responses = responses
        self.error_types = defaultdict(int)
        self.contexts = defaultdict(int)

    def analyze(self):
        for response in self.responses:
            # Extract error type and context
            error_type_match = re.search(r"Error: (\w+)", response)
            context_match = re.search(r"Context: (.+?)\n", response)
            
            if error_type_match:
                error_type = error_type_match.group(1)
                self.error_types[error_type] += 1
            if context_match:
                context = context_match.group(1)
                self.contexts[context] += 1

            # Print progress
            print(f"Analyzing response: {len(self.responses) - self.responses.index(response)}")

        # Generate and print report
        print("\nError Types:")
        for error_type, count in self.error_types.items():
            print(f"{error_type}: {count}")

        print("\nContexts:")
        for context, count in self.contexts.items():
            print(f"{context}: {count}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python error_analyzer.py <response_file>")
        exit(1)

    with open(sys.argv[1], "r") as file:
        responses = [line.strip() for line in file.readlines()]
        analyzer = ErrorAnalyzer(responses)
        analyzer.analyze()

if __name__ == "__main__":
    main()