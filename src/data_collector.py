import requests
import pandas as pd 
import time 
import os

#CONFIGURATION

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}"
} if GITHUB_TOKEN else {}

REPOS = [
    ("pallets", "flask"),
    ("tiangolo", "fastapi"),
    ("django", "django"),
    ("numpy", "numpy"),
    ("pytorch", "pytorch"),
    ("tensorflow", "tensorflow"),
    ("scikit-learn", "scikit-learn"),
    ("keras-team", "keras"),
    ("microsoft", "vscode"),
    ("facebook", "react"),
    ("huggingface", "transformers"),
    ("langchain-ai", "langchain")
]

OUTPUT_FILE = "data/github_commits_large.csv"

COMMITS_PER_PAGE = 100
MAX_PAGES = 30   # Increase for more data


# =========================
# RATE LIMIT HANDLER
# =========================

def handle_rate_limit(response):

    if response.status_code == 403:

        print("Rate limit hit. Sleeping for 60 seconds...")
        time.sleep(60)
        return True

    return False


# =========================
# FETCH COMMITS
# =========================

def fetch_repo_commits(owner, repo):

    print(f"\nCollecting from {owner}/{repo}")

    commits = []

    for page in range(1, MAX_PAGES + 1):

        url = f"https://api.github.com/repos/{owner}/{repo}/commits"

        params = {
            "per_page": COMMITS_PER_PAGE,
            "page": page
        }

        response = requests.get(url, headers=HEADERS, params=params)

        if handle_rate_limit(response):
            response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code != 200:
            print(f"Error fetching page {page}")
            break

        data = response.json()

        if not data:
            break

        for commit in data:

            try:

                author = commit["commit"]["author"]

                commits.append({

                    "repo": repo,
                    "developer": author["name"],
                    "date": author["date"],
                    "message_length": len(commit["commit"]["message"]),
                    "commit_id": commit["sha"]

                })

            except:
                continue

        print(f"Page {page} collected")

        time.sleep(0.2)

    return commits


# =========================
# FEATURE ENGINEERING
# =========================

def engineer_features(df):

    print("\nEngineering features...")

    df["date"] = pd.to_datetime(df["date"])

    df["commit_hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # heuristic productivity risk
    df["risk_label"] = df["message_length"].apply(
        lambda x: 1 if x < 15 else 0
    )

    return df


# =========================
# MAIN PIPELINE
# =========================

def main():

    os.makedirs("data", exist_ok=True)

    all_commits = []

    for owner, repo in REPOS:

        repo_commits = fetch_repo_commits(owner, repo)

        all_commits.extend(repo_commits)

        print(f"Total commits collected so far: {len(all_commits)}")

    df = pd.DataFrame(all_commits)

    df = engineer_features(df)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset collection complete.")
    print(f"Total entries: {len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()