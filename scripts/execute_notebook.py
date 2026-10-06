"""
Executes the notebook in-place using nbclient and saves all generated cell outputs.
"""

import nbformat
from nbclient import NotebookClient

def execute():
    nb_path = "notebooks/Explainable_Breast_Cancer_Classification.ipynb"
    print(f"Reading notebook: {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    print("Executing all cells (this will run EDA, 2-Stage Selection, Model Benchmarking, Threshold Calibration, and SHAP)...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    
    print("Execution complete. Saving rendered notebook with all outputs...")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print(f"Notebook '{nb_path}' successfully executed and saved with all figures, tables, and metrics!")

if __name__ == "__main__":
    execute()
