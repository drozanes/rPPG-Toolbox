import os

dir_path = "neural_methods/trainer"
for filename in os.listdir(dir_path):
    if filename.endswith("Trainer.py"):
        filepath = os.path.join(dir_path, filename)
        with open(filepath, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        has_wandb = False
        has_sys = False
        
        for line in lines:
            if "import wandb" in line: has_wandb = True
            if "import sys" in line: has_sys = True
            
        if not has_wandb:
            new_lines.append("try:\n    import wandb\nexcept ImportError:\n    pass\n")
        if not has_sys:
            new_lines.append("import sys\n")
            
        for line in lines:
            new_lines.append(line)
            stripped = line.strip()
            if stripped.startswith("mean_training_losses.append("):
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}try:\n{indent}    if 'wandb' in sys.modules:\n{indent}        wandb.log({{'Train Loss': mean_training_losses[-1], 'Epoch': epoch}})\n{indent}except Exception:\n{indent}    pass\n")
            elif stripped.startswith("mean_valid_losses.append("):
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}try:\n{indent}    if 'wandb' in sys.modules:\n{indent}        wandb.log({{'Validation Loss': mean_valid_losses[-1], 'Epoch': epoch}})\n{indent}except Exception:\n{indent}    pass\n")
                
        with open(filepath, "w") as f:
            f.writelines(new_lines)
        print(f"Patched {filename}")
