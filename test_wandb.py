import wandb
api = wandb.Api()
runs = api.runs("rPPG-Master-Thesis")
print(f"Found {len(runs)} runs")
for run in runs[:1]:
    print(f"Run {run.name} ID {run.id}")
    print("Config:", run.config)
    
    # Try to get logs
    history = run.history()
    print("History keys:", history.keys())
    
    # Try to get files (terminal output)
    files = run.files()
    for f in files:
        if f.name == 'output.log':
            print("Found output.log, downloading...")
            f.download(replace=True)
            with open('output.log') as f_log:
                print(f_log.read()[:500])
