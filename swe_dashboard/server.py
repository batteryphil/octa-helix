import os
import glob
import json
from flask import Flask, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

TRAJECTORY_BASE_DIR = "/home/phil/.gemini/antigravity/scratch/analysis_project/SWE-agent/trajectories"

def get_trajectories():
    """Scans the base directory for all .traj or .info.log files."""
    traj_files = glob.glob(os.path.join(TRAJECTORY_BASE_DIR, "**", "*.traj"), recursive=True)
    log_files = glob.glob(os.path.join(TRAJECTORY_BASE_DIR, "**", "*.info.log"), recursive=True)
    
    # Combine and deduplicate by directory
    all_files = set(traj_files)
    for log in log_files:
        traj_guess = log.replace(".info.log", ".traj")
        if traj_guess not in all_files:
            all_files.add(traj_guess)
            
    results = []
    for f in list(all_files):
        if "demonstrations" in f:
            continue
            
        # Parse the directory structure to get useful info
        parts = f.split(os.sep)
        file_name = parts[-1]
        
        # Determine status (running, error, done) by checking log files
        dir_name = os.path.dirname(f)
        info_log = os.path.join(dir_name, file_name.replace(".traj", ".info.log"))
        
        status = "Unknown"
        if os.path.exists(info_log):
            with open(info_log, 'r') as log_f:
                content = log_f.read()
                
                # Find the last occurrence of key events
                retry_idx = content.rfind("Retrying LM query")
                init_idx = content.rfind("Environment Initialized")
                done_idx = content.rfind("DONE")
                
                if done_idx > retry_idx and done_idx > init_idx:
                    status = "Completed"
                elif retry_idx > init_idx:
                    status = "Error/Retrying"
                elif init_idx != -1:
                    status = "Active"
                    
        # Use directory modification time since traj file might not exist yet
        mtime = os.path.getmtime(os.path.dirname(f)) if os.path.exists(os.path.dirname(f)) else 0
        
        results.append({
            "id": file_name,
            "path": f,
            "name": file_name.replace(".traj", ""),
            "status": status,
            "mtime": mtime
        })
        
    # Sort by modification time
    results.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/trajectories")
def list_trajectories():
    return jsonify(get_trajectories())

@app.route("/api/stats")
def get_stats():
    submitted_dir = "/home/phil/.gemini/antigravity/scratch/analysis_project/bounty-helix/solutions/submitted"
    pr_count = 0
    if os.path.exists(submitted_dir):
        pr_count = len([d for d in os.listdir(submitted_dir) if os.path.isdir(os.path.join(submitted_dir, d))])
    
    return jsonify({"pr_count": pr_count})

@app.route("/api/trajectory/<path:traj_id>")
def get_trajectory(traj_id):
    # Find the requested file
    trajectories = get_trajectories()
    target_file = next((t["path"] for t in trajectories if t["id"] == traj_id), None)
    
    if not target_file or not os.path.exists(target_file):
        return jsonify({"steps": []})  # Return empty steps if file not written yet
        
    steps = []
    try:
        with open(target_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        steps.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    return jsonify({"steps": steps})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
