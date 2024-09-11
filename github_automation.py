import os
import subprocess
import pandas as pd

def push_csv_to_github():
    today = pd.to_datetime('today').strftime('%Y-%m-%d')
    file_path = f"versions\\updated_aov_{today}.csv"
    
    # Stage the CSV file
    subprocess.run(['git', 'add', file_path], check=True)

    # Commit the changes
    subprocess.run(['git', 'commit', '-m', f"updated_aov_{today}.csv successfully pushed!"], check=True)

    # Push the changes to the specified branch
    push_cmd = ['git', 'push', 'origin', 'main']
    subprocess.run(push_cmd, check=True)

    print(f'Successfully pushed {file_path} to GitHub on branch main')

push_csv_to_github()