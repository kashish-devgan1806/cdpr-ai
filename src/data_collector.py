import requests
import pandas as pd 
import time 
import os
from dotenv import load_dotenv

load_dotenv()

#CONFIGURATION

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    print("WARNING: No GitHub token found. Rate limits will be very low.")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
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


# RATE LIMIT HANDLER

def handle_rate_limit(response):

    if response.status_code == 403:

        print("Rate limit hit. Sleeping for 10 seconds...")
        time.sleep(10)
        return True

    return False


# FETCH COMMITS

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


# FEATURE ENGINEERING

def engineer_features(df):

    print("\nEngineering features...")

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Basic time features
    df["commit_hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # =========================
    # BEHAVIORAL BASELINE PER DEVELOPER
    # =========================

    dev_stats = df.groupby("developer")["commit_hour"].agg(
        dev_mean_hour="mean",
        dev_std_hour="std"
    ).reset_index()

    # Fill NaN std (happens if developer has only 1 commit)
    dev_stats["dev_std_hour"] = dev_stats["dev_std_hour"].fillna(0)

    # Merge back
    df = df.merge(dev_stats, on="developer", how="left")

    # =========================
    # DEVIATION FEATURE
    # =========================

    df["hour_deviation"] = abs(df["commit_hour"] - df["dev_mean_hour"])

    # =========================
    # REALISTIC RISK LABEL (NON-DETERMINISTIC)
    # =========================

    df["risk_label"] = (
        (df["hour_deviation"] > df["dev_std_hour"]) &
        (df["dev_std_hour"] > 0)
    ).astype(int)

    print("Features engineered successfully.")

    return df


# MAIN PIPELINE

def main():
    os.makedirs("data", exist_ok=True)
    temp_file = "data/temp_commits.csv"
    all_commits = []

    # Resume if temp file exists
    if os.path.exists(temp_file):
        try:
            print("Resuming from saved progress...")
            temp_df = pd.read_csv(temp_file)

            if not temp_df.empty:
                all_commits = temp_df.to_dict("records")
            else:
                print("Temp file empty. Starting fresh.")

        except Exception as e:
            print("Temp file corrupted. Starting fresh.")
            all_commits = []

    for owner, repo in REPOS:
        repo_commits = fetch_repo_commits(owner, repo)
        all_commits.extend(repo_commits)

        # Autosave every repo
        pd.DataFrame(all_commits).to_csv(temp_file, index=False)
        print(f"Autosaved progress: {len(all_commits)} commits")

    df = pd.DataFrame(all_commits)
    df = engineer_features(df)
    df.to_csv("data/github_commits_large.csv", index=False)
    print(f"\nFinal dataset size: {len(df)}")


if __name__ == "__main__":
    main()