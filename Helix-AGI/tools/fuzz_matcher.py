import fuzzywuzzy
from fuzzywuzzy import process

def find_best_match(query, strings):
    return process.extractOne(query, strings, scorer=fuzzywuzzy.partial_ratio)

if __name__ == '__main__':
    query = "example query"
    strings = ["example string 1", "example string 2", "example string 3"]
    best_match, score = find_best_match(query, strings)
    print(f"Best match: {best_match} (score: {score})")