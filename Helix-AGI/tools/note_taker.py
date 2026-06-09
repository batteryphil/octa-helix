"""
A simple note-taking tool to capture and organize ideas.

Usage:
1. Create a new note: nt = Note(title="My Note", content="This is my first note.")
2. Save a note: nt.save()
3. Retrieve a note: nt = Note.load("My Note")
4. List all notes: Note.list_all()
5. Search for a note: nt = Note.search("first")

Note: Notes are saved in a JSON file called 'notes.json'.
"""

import json
import os

class Note:
    _notes = []

    def __init__(self, title, content=""):
        self.title = title
        self.content = content
        self._notes.append(self)

    @classmethod
    def save(cls):
        with open("notes.json", "w") as f:
            json.dump([note.__dict__ for note in cls._notes], f)

    @classmethod
    def load(cls, title):
        if os.path.exists("notes.json"):
            with open("notes.json", "r") as f:
                loaded_notes = json.load(f)
                for note_dict in loaded_notes:
                    note = Note(**note_dict)
                    if note.title == title:
                        return note
        return None

    @classmethod
    def list_all(cls):
        if os.path.exists("notes.json"):
            with open("notes.json", "r") as f:
                loaded_notes = json.load(f)
                for note_dict in loaded_notes:
                    print(f"Title: {note_dict['title']}\nContent: {note_dict['content']}\n")

    @classmethod
    def search(cls, keyword):
        if os.path.exists("notes.json"):
            with open("notes.json", "r") as f:
                loaded_notes = json.load(f)
                for note_dict in loaded_notes:
                    if keyword in note_dict["content"]:
                        print(f"Title: {note_dict['title']}\nContent: {note_dict['content']}\n")