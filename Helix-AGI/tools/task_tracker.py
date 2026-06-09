"""
A simple task tracker that allows users to set, update, and view their personal goals or tasks.
"""

import json
import os
from dataclasses import dataclass, fields
from typing import Any, List

import toolset

@dataclass
class Task:
    name: str
    description: str
    due_date: str
    completed: bool = False

    def __str__(self) -> str:
        return f"{self.name} - {self.description} (Due: {self.due_date}, Completed: {'Yes' if self.completed else 'No'})"

class TaskTracker:
    def __init__(self):
        self.tasks: List[Task] = []

    def add_task(self, task: Task):
        self.tasks.append(task)

    def update_task(self, index: int, task: Task):
        if 0 <= index < len(self.tasks):
            self.tasks[index] = task
        else:
            raise IndexError("Index out of range")

    def remove_task(self, index: int):
        if 0 <= index < len(self.tasks):
            del self.tasks[index]
        else:
            raise IndexError("Index out of range")

    def view_tasks(self):
        return self.tasks

    def save(self, filename: str):
        with open(filename, 'w') as f:
            json.dump([dict(field.name for field in fields(Task)), *[
                dict(to_dict(task) for task in self.tasks)
            ]], f, indent=2)

    def load(self, filename: str):
        if not os.path.exists(filename):
            return
        with open(filename, 'r') as f:
            data = json.load(f)
            self.tasks = [Task(**{field.name: value for field, value in field.items()} ) for field in data[1:]]

    def to_dict(self, task: Task) -> dict:
        return {
            field.name: getattr(task, field.name) for field in fields(Task)
        }

    def from_dict(cls, data: dict) -> Task:
        return Task(**data)

toolset.register_tool('task_tracker', TaskTracker)