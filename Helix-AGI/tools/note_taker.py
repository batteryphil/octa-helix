"""
A simple note-taking tool that allows users to save notes with titles and content, tag them, and search by tag.
"""

import json
import os

class NoteTaker:
    def __init__(self, toolset='self'):
        self.toolset = toolset
        self.notes = {}
        self.tags = {}
        self.load_notes()

    def load_notes(self):
        if os.path.exists(f"{self.toolset}_notes.json"):
            with open(f"{self.toolset}_notes.json", "r") as f:
                self.notes = json.load(f)
                for note_id, note in self.notes.items():
                    self.tags[note_id] = note['tags']

    def save_notes(self):
        with open(f"{self.toolset}_notes.json", "w") as f:
            json.dump(self.notes, f)

    def add_note(self, title, content, tags):
        note_id = len(self.notes) + 1
        self.notes[note_id] = {'title': title, 'content': content, 'tags': tags}
        self.tags[note_id] = tags
        self.save_notes()
        return note_id

    def get_note(self, note_id):
        if note_id in self.notes:
            return self.notes[note_id]
        else:
            return None

    def update_note(self, note_id, title=None, content=None, tags=None):
        if note_id in self.notes:
            if title:
                self.notes[note_id]['title'] = title
            if content:
                self.notes[note_id]['content'] = content
            if tags:
                self.notes[note_id]['tags'] = tags
                self.tags[note_id] = tags
            self.save_notes()

    def delete_note(self, note_id):
        if note_id in self.notes:
            del self.notes[note_id]
            del self.tags[note_id]
            self.save_notes()

    def get_notes_by_tag(self, tag):
        return {note_id: note for note_id, note in self.notes.items() if tag in note['tags']}

def main():
    note_taker = NoteTaker(toolset='helix')
    note_taker.add_note("Meeting Minutes", "Discussed project timeline and milestones.", ["meeting", "project"])
    note_taker.add_note("Book Recommendations", "Read 'The Great Gatsby' and enjoyed it.", ["books", "recommendations"])
    print(note_taker.get_notes_by_tag("meeting"))
    note_taker.update_note(1, title="Meeting Notes", tags=["meeting"])
    print(note_taker.get_note(1))

if __name__ == "__main__":
    main()