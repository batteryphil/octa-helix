"""
A simple task tracker that allows you to manage tasks with descriptions, due dates, and progress markers.
"""

import json
import datetime
from collections import defaultdict

class TaskTracker:
    def __init__(self):
        self.tasks = []

    def add_task(self, description, due_date=None):
        task = {
            "id": len(self.tasks),
            "description": description,
            "due_date": due_date,
            "completed": False
        }
        self.tasks.append(task)
        return task

    def update_task(self, task_id, description=None, due_date=None, completed=None):
        task = next((t for t in self.tasks if t["id"] == task_id), None)
        if task:
            if description:
                task["description"] = description
            if due_date:
                task["due_date"] = due_date
            if completed is not None:
                task["completed"] = completed
            return task
        return None

    def complete_task(self, task_id):
        task = next((t for t in self.tasks if t["id"] == task_id), None)
        if task:
            task["completed"] = True
            return task
        return None

    def delete_task(self, task_id):
        self.tasks = [t for t in self.tasks if t["id"] != task_id]

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.tasks, f)

    def load(self, filename):
        with open(filename, "r") as f:
            self.tasks = json.load(f)

def main():
    tracker = TaskTracker()
    tracker.add_task("Buy groceries", due_date="2023-04-01")
    tracker.add_task("Finish project report", due_date="2023-04-15")
    tracker.update_task(0, description="Buy milk, eggs, bread")
    tracker.complete_task(0)
    tracker.save("tasks.json")
    tracker.load("tasks.json")
    print(tracker.tasks)

if __name__ == "__main__":
    main()