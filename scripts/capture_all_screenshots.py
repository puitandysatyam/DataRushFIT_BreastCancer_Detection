"""
Captures high-resolution screenshots for all 7 dashboard pages using headless Edge.
Saves to screenshots/
"""

import os
import subprocess
import time

tabs = [
    ("page1_clinical_signal.png", "tab-signal"),
    ("page2_feature_lab.png", "tab-featurelab"),
    ("page3_model_bench.png", "tab-modelbench"),
    ("page4_why_this_prediction.png", "tab-explainability"),
    ("page5_patient_explorer.png", "tab-patients"),
    ("page6_error_map.png", "tab-errormap"),
    ("page7_method_and_data.png", "tab-method"),
]

edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge_exe):
    edge_exe = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

base_dir = os.path.abspath(".").replace("\\", "/")
os.makedirs("screenshots", exist_ok=True)

for filename, tab_id in tabs:
    out_path = os.path.abspath(f"screenshots/{filename}")
    url = f"file:///{base_dir}/frontend/index.html?tab={tab_id}"
    print(f"Capturing {filename} from {url}...")
    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        f"--screenshot={out_path}",
        "--window-size=1440,900",
        url
    ]
    subprocess.run(cmd)
    time.sleep(1)

print("All 7 page screenshots captured successfully!")

