import json
from typing import List, Dict

def belief_conflict_resolution(beliefs: List[Dict[str, float]]) -> Dict[str, float]:
    # Sort beliefs by confidence score in descending order
    sorted_beliefs = sorted(beliefs, key=lambda x: x['confidence'], reverse=True)
    
    # Initialize variables
    consensus_belief = {}
    min_conflict = float('inf')
    
    # Iterate through the sorted beliefs to find the consensus belief
    for belief in sorted_beliefs:
        current_conflict = 0
        for other_belief in sorted_beliefs:
            if belief != other_belief:
                # Calculate the conflict between the current belief and other beliefs
                conflict = abs(belief['confidence'] - other_belief['confidence'])
                current_conflict += conflict
        
        # Check if the current belief has the lowest conflict
        if current_conflict < min_conflict:
            min_conflict = current_conflict
            consensus_belief = belief
    
    return consensus_belief