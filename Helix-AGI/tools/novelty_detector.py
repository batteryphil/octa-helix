"""
A tool to identify and track novel beliefs formed during conversation.

This tool analyzes conversation context and queries a knowledge graph to identify new facts or insights that are not present in existing memories.
"""

import json
import spacy
from knowledge_graph import KnowledgeGraph

class NoveltyDetector:
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.nlp = spacy.load("en_core_web_sm")

    def process_conversation(self, conversation):
        novel_facts = []
        for utterance in conversation:
            doc = self.nlp(utterance)
            for token in doc:
                if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                    fact = " ".join([str(token.text) for token in doc if token.dep_ == "nsubj"] + [str(token.text) for token in doc if token.dep_ == "dobj"])
                    if not self.knowledge_graph.has_fact(fact):
                        novel_facts.append(fact)
        return novel_facts

    def run(self, conversation):
        novel_facts = self.process_conversation(conversation)
        return json.dumps(novel_facts)

def main():
    knowledge_graph = KnowledgeGraph()
    novelty_detector = NoveltyDetector(knowledge_graph)
    conversation = [
        "I ate an apple.",
        "Apples are healthy.",
        "I saw a dog.",
        "Dogs are friendly."
    ]
    print(novelty_detector.run(conversation))

if __name__ == "__main__":
    main()