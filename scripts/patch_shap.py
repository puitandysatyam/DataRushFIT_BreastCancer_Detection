import sys

tree_path = r"C:\Users\VICTUS\AppData\Local\Programs\Python\Python310\lib\site-packages\shap\explainers\_tree.py"
with open(tree_path, "r", encoding="utf-8") as f:
    content = f.read()

target1 = 'self.base_score = float(learner_model_param["base_score"])'
rep1 = '''bs_val = learner_model_param["base_score"]
        bs_clean = bs_val.strip("[]") if isinstance(bs_val, str) else bs_val
        self.base_score = float(bs_clean)'''

target2 = 'base_score = float(learner_model_param["base_score"])'
rep2 = 'base_score = self.base_score'

if target1 in content and target2 in content:
    content = content.replace(target1, rep1, 1).replace(target2, rep2, 1)
    with open(tree_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESSFULLY PATCHED SHAP!")
else:
    print("Target not found, checking if already patched...")
    if 'bs_clean' in content:
        print("ALREADY PATCHED!")
    else:
        print("Not patched and not found.")

