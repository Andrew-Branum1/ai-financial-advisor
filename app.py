# app.py
from flask import Flask, render_template, jsonify
import os
import logging
import subprocess
import json

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Path to the directory where plots and reports are saved
OUTPUT_DIR = os.path.join(os.getcwd(), 'plots')

@app.route('/')
def home():
    """
    Renders the main dashboard page.
    """
    # Pass a list of available reports to the template
    # This assumes a naming convention like '..._llm_report.txt'
    try:
        files = os.listdir(OUTPUT_DIR)
        reports = sorted([f for f in files if f.endswith('_llm_report.txt')], reverse=True)
    except FileNotFoundError:
        reports = []
    
    return render_template('index.html', reports=reports)


@app.route('/run_evaluation')
def run_evaluation():
    """
    API endpoint to trigger a new evaluation run.
    This runs your rl/evaluate.py script as a separate process.
    """
    logging.info("Received request to run evaluation...")
    try:
        # We use subprocess to run your existing script.
        # This is a simple way to integrate without major refactoring.
        # Make sure your Python environment is activated or specify the full path to the python executable.
        process = subprocess.run(
            ['python', '-m', 'rl.evaluate'],
            capture_output=True,
            text=True,
            check=True,  # This will raise an exception if the script fails
            encoding='utf-8'
        )
        logging.info("Evaluation script completed successfully.")
        logging.info(f"Script output:\n{process.stdout}")
        return jsonify({"status": "success", "message": "Evaluation completed successfully. Refresh to see new report."})
    except subprocess.CalledProcessError as e:
        logging.error(f"Evaluation script failed with exit code {e.returncode}.")
        logging.error(f"Error output:\n{e.stderr}")
        return jsonify({"status": "error", "message": "Evaluation script failed.", "details": e.stderr}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred while running the evaluation script: {e}")
        return jsonify({"status": "error", "message": "An unexpected server error occurred."}), 500


@app.route('/get_report/<report_name>')
def get_report(report_name):
    """
    Fetches the content of a specific report and its associated plots.
    """
    # Basic security check
    if '..' in report_name or not report_name.endswith('_llm_report.txt'):
        return jsonify({"error": "Invalid report name"}), 400

    base_name = report_name.replace('_llm_report.txt', '')
    
    try:
        with open(os.path.join(OUTPUT_DIR, report_name), 'r') as f:
            report_text = f.read()
            
        # Find associated plots
        plots = {
            "performance": f"/static/plots/{base_name}_performance.png",
            "weights": f"/static/plots/{base_name}_weights.png",
            "distribution": f"/static/plots/{base_name}_returns_distribution.png"
        }
        
        # Check if plot files exist
        for key, path in plots.items():
            if not os.path.exists(os.path.join(os.getcwd(), path.replace('/static/', ''))):
                 plots[key] = None # Set to None if plot doesn't exist

        return jsonify({
            "report_title": base_name,
            "report_text": report_text,
            "plots": plots
        })

    except FileNotFoundError:
        return jsonify({"error": "Report not found"}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5001)

