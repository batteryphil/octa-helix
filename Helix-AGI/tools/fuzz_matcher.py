import fuzzywuzzy
from fuzzywuzzy import process

def find_best_match(strings, target):
    return process.extractOne(target, strings, scorer=fuzzywuzzy.partial_ratio)

if __name__ == '__main__':
    strings = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    target = 'banana'
    best_match, score = find_best_match(strings, target)
    print(f"Best match: {best_match} (score: {score})")