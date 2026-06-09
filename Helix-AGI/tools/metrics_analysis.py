"""
A tool to analyze performance metrics and provide insights for self-improvement.

This tool reads performance data files, calculates relevant metrics, and outputs a report highlighting areas where the user is excelling and where they need to focus their efforts.
"""

import json
import os
import sys
import statistics
import ToolRegistry

class MetricsAnalysis:
    def __init__(self, data_files):
        self.data_files = data_files
    
    def read_data(self):
        data = []
        for file in self.data_files:
            with open(file, 'r') as f:
                data.append(json.load(f))
        return data
    
    def calculate_metrics(self, data):
        metrics = {
            'total_tasks': len(data),
            'total_time': sum(task['time'] for task in data),
            'average_time': statistics.mean(task['time'] for task in data),
            'median_time': statistics.median(task['time'] for task in data),
            'longest_task': max(task['time'] for task in data),
            'shortest_task': min(task['time'] for task in data),
            'tasks_completed_today': sum(1 for task in data if task['date'] == 'today'),
            'tasks_completed_yesterday': sum(1 for task in data if task['date'] == 'yesterday'),
        }
        return metrics
    
    def generate_report(self, metrics):
        report = f"Performance Metrics Report\n"
        report += f"Total tasks: {metrics['total_tasks']}\n"
        report += f"Total time: {metrics['total_time']} seconds\n"
        report += f"Average time per task: {metrics['average_time']} seconds\n"
        report += f"Median time per task: {metrics['median_time']} seconds\n"
        report += f"Longest task: {metrics['longest_task']} seconds\n"
        report += f"Shortest task: {metrics['shortest_task']} seconds\n"
        report += f"Tasks completed today: {metrics['tasks_completed_today']}\n"
        report += f"Tasks completed yesterday: {metrics['tasks_completed_yesterday']}\n"
        return report
    
    def run(self):
        data = self.read_data()
        metrics = self.calculate_metrics(data)
        report = self.generate_report(metrics)
        print(report)

def main():
    if len(sys.argv) < 2:
        print("Usage: python metrics_analysis.py <data_file1> <data_file2> ...")
        sys.exit(1)
    
    data_files = sys.argv[1:]
    metrics_analysis = MetricsAnalysis(data_files)
    metrics_analysis.run()

if __name__ == '__main__':
    ToolRegistry.register_tool('metrics_analysis', 'self', MetricsAnalysis)
    main()