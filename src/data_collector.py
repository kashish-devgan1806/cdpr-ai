import requests
import pandas as pd 
import time 
import os

#CONFIGURATION

REPOS = [
    ("pallets","flask"),
    ("tiangolo","fastapi"),
    ("scikit-learn","scikit-learn")
]

OUTPUT_PATH = "data/github_tasks.csv"

#GITHUB API LOGIC 

def fetch_commits(owner, repo, max_pages=5):
    print(f"Fetching commits from {owner}/{repo}")
    
    commits = []
    base_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    
    for page in range(1, max_pages + 1):
        url = f"{base_url}?page={page}&per_page=50"
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch commits for {owner}/{repo} (status code: {response.status_code})")
            break
        
        data = response.json()
        
        if not data:
            break
        
        for commit in data:
            author = commit["commit"]["author"]
            
            commits.append({
                "repo": repo,
                "developer": author["name"],
                "date": author["date"],
                "message_length": len(commit["commit"]["message"]),
            })
            
        time.sleep(0.5)  
    return commits

#FEATURE ENGINEERING

def create_dataset(all_commits):
    df = pd.DataFrame(all_commits)
    df["date"] = pd.to_datetime(df["date"])
    
    commit_counts = df.groupby(["repo", "developer",]).size().reset_index(name = "total_commits")
    
    msg_length = df.groupby(["repo", "developer",])["message_length"].mean().reset_index()
    
    dataset = commit_counts.merge(msg_length, on=["repo", "developer"])
    
    dataset["risk_label"] = dataset["total_commits"].apply(
        lambda x: "HIGH" if x <= 3 else "LOW"
    )
    
    return dataset

#MAIN LOGIC

def main():
    os.makedirs("data", exist_ok=True)
    all_commits = []
    for owner, repo in REPOS:
        commits = fetch_commits(owner, repo)
        all_commits.extend(commits)
        
    dataset = create_dataset(all_commits)
    dataset.to_csv(OUTPUT_PATH, index=False)
    
    print("\nDATASET CREATED SUCCESFULLY!")
    print(f"Saved to {OUTPUT_PATH}")
    print("\nPreview:")
    print(dataset.head())
    
    
if __name__ == "__main__":
    main()