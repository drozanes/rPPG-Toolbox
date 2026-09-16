import wandb
import re
import os

api = wandb.Api()
runs = api.runs("rPPG-Master-Thesis")
print(f"Found {len(runs)} runs")

for run in runs:
    # Check if run name looks like default wandb generated names (e.g. random adjectives/nouns)
    # Actually, we can just rename all of them if we find the config, or only if they are not already renamed.
    # Let's rename all runs that don't match our new format.
    
    if run.name.count('_') >= 3 and 'lr' in run.name and 'bs' in run.name:
        print(f"Skipping {run.name} (already renamed)")
        continue

    print(f"Processing run {run.name} ({run.id})...")
    
    # Download output.log
    files = run.files()
    has_log = False
    for f in files:
        if f.name == 'output.log':
            f.download(replace=True)
            has_log = True
            break
            
    if not has_log or not os.path.exists('output.log'):
        print(f"  No output.log found for {run.name}, skipping.")
        continue
        
    with open('output.log', 'r') as f:
        content = f.read()
        
    # Extract
    model = "UnkModel"
    match = re.search(r'MODEL:\n.*?NAME:\s*([^\n]+)', content, re.DOTALL)
    if match:
        model = match.group(1).strip()
    elif "NAME:" in content:
        for line in content.split('\n'):
            if "  NAME:" in line:
                model = line.split("NAME:")[1].strip()
                break
                
    dataset = "UnkDataset"
    match = re.search(r'TRAIN:\n.*?DATA:\n.*?DATASET:\s*([^\n]+)', content, re.DOTALL)
    if match:
        dataset = match.group(1).strip()
    else:
        for line in content.split('\n'):
            if "    DATASET:" in line:
                val = line.split("DATASET:")[1].strip()
                if val:
                    dataset = val
                    break
                    
    lr = "UnkLR"
    bs = "UnkBS"
    epochs = "UnkEp"
    
    for line in content.split('\n'):
        if line.startswith("  LR: "):
            lr = line.split("LR: ")[1].strip()
        elif line.startswith("  BATCH_SIZE: "):
            bs = line.split("BATCH_SIZE: ")[1].strip()
        elif line.startswith("  EPOCHS: "):
            epochs = line.split("EPOCHS: ")[1].strip()
            
    new_name = f"{dataset}_{model}_lr{lr}_bs{bs}_ep{epochs}"
    
    print(f"  Renaming {run.name} -> {new_name}")
    run.name = new_name
    run.update()

if os.path.exists('output.log'):
    os.remove('output.log')
