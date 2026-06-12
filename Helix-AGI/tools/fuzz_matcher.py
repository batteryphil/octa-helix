import fuzzywuzzy
from fuzzywuzzy import process

def fuzz_match(query, candidates):
    return process.extractOne(query, candidates, scorer=fuzzywuzzy.partial_ratio)

if __name__ == '__main__':
    query = "apple"
    candidates = ["apple", "banana", "orange", "grape"]
    match, score = fuzz_match(query, candidates)
    print(f"Best match: {match} (score: {score})")