"""
This module provides a tool for optimizing and pruning beliefs based on performance metrics.

The belief_optimizer tool analyzes performance metrics to identify beliefs that are not aligned with successful outcomes. It suggests modifications to these beliefs or flags them for review.

To use this tool, you need to provide a list of beliefs and their corresponding performance metrics. The tool will then process the data and generate suggestions for belief optimization.
"""

import json
from typing import List, Dict
from helix.registry import ToolRegistry

class BeliefOptimizer:
    """
    A tool for optimizing and pruning beliefs based on performance metrics.
    """
    
    def __init__(self, beliefs: List[Dict]):
        """
        Initialize the BeliefOptimizer with a list of beliefs and their performance metrics.
        
        :param beliefs: A list of dictionaries, where each dictionary represents a belief and its performance metrics.
                        Each dictionary should have the following structure:
                        {
                            "belief": The belief statement,
                            "metric": The performance metric associated with the belief,
                            "value": The value of the performance metric
                        }
        """
        self.beliefs = beliefs
    
    def optimize(self) -> List[Dict]:
        """
        Analyze the beliefs and their performance metrics to identify beliefs that are not aligned with successful outcomes.
        Suggest modifications to these beliefs or flag them for review.
        
        :return: A list of dictionaries, where each dictionary represents a belief and its optimization suggestion.
                 Each dictionary should have the following structure:
                 {
                     "belief": The belief statement,
                     "suggestion": The optimization suggestion for the belief
                 }
        """
        suggestions = []
        for belief in self.beliefs:
            if belief["value"] < 0.5:  # Assuming a threshold of 0.5 for successful outcomes
                suggestions.append({
                    "belief": belief["belief"],
                    "suggestion": "Review and modify the belief to align with successful outcomes"
                })
            else:
                suggestions.append({
                    "belief": belief["belief"],
                    "suggestion": "The belief is aligned with successful outcomes"
                })
        return suggestions

def main():
    # Example usage
    beliefs = [
        {"belief": "I can achieve any goal I set my mind to", "metric": "goal_achievement", "value": 0.8},
        {"belief": "I am not good at public speaking", "metric": "public_speaking", "value": 0.2},
        {"belief": "I am capable of learning new skills quickly", "metric": "learning_ability", "value": 0.9}
    ]
    
    optimizer = BeliefOptimizer(beliefs)
    suggestions = optimizer.optimize()
    
    print(json.dumps(suggestions, indent=2))

if __name__ == "__main__":
    ToolRegistry.register_tool("belief_optimizer", BeliefOptimizer)
    main()