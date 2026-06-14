import fuzzywuzzy
from fuzzywuzzy import process

def fuzz_match(query, options):
    return process.extractOne(query, options, scorer=fuzzywuzzy.partial_ratio)

if __name__ == '__main__':
    query = "apple"
    options = ["apple", "banana", "orange", "grape"]
    best_match, score = fuzz_match(query, options)
    print(f"Best match: {best_match}, Score: {score}")