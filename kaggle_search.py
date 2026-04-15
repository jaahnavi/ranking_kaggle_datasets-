"""
Read gaps.json and fetch Kaggle datasets & notebooks
Run this after step1_extract_gaps.py has created gaps.json

Requirements:
    pip install kaggle
    ~/.kaggle/kaggle.json must exist with your API key
"""

import re
import json
import numpy as np
import kaggle

kaggle.api.authenticate()
api = kaggle.api

# CONFIG

GAPS_INPUT_FILE = "gaps.json"
REPORT_OUTPUT_FILE = "kaggle_report.json"
KAGGLE_RESULTS_PER_SKILL = 10


# LOAD gaps.json
def load_gaps(path: str = GAPS_INPUT_FILE) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        gaps = json.load(f)
    print(f"📂  Loaded gaps from: {path}")
    print(f"    Missing skills: {gaps['missing_skills']}\n")
    return gaps



# SEARCH TERM CLEANER
def clean_skill_for_search(skill: str) -> str:
    """Strip parenthetical examples and keep only the first 3 words."""
    skill = re.sub(r'\(.*?\)', '', skill)   # remove (Prophet, ARIMA) etc.
    skill = re.sub(r'\s+', ' ', skill)      # collapse whitespace
    words = skill.strip().split()
    return ' '.join(words[:3])


# KAGGLE HELPERS
def fetch_kaggle_datasets(skill: str, limit: int = KAGGLE_RESULTS_PER_SKILL) -> list:
    try:
        query = clean_skill_for_search(skill)
        print(f"       → searching: '{query}'")
        datasets = api.dataset_list(search=query, sort_by="votes")
        
        ## DEBUG
        print(f"       → raw result count: {len(datasets)}")
        if datasets:
            print(f"       → first result: {datasets[0].title}")
       
        results = []
        for ds in datasets[:limit]:
            results.append({
                "title":        ds.title,
                "url":          f"https://www.kaggle.com/datasets/{ds.ref}",
                "description":  ds.subtitle,
                "votes":        ds.vote_count,
                "downloads":    ds.download_count,
                "size":         ds.total_bytes,
                "usability":    round(ds.usability_rating, 2) if ds.usability_rating else None,
                "last_updated": str(ds.last_updated)[:10],
                "license":      ds.license_name,
                "tags":         [t.name for t in ds.tags] if ds.tags else [],
                "author":       ds.creator_name,
            })
        return results
    except Exception as e:
        print(f"    ⚠️  Dataset search failed for '{skill}': {e}")
        return []


def fetch_kaggle_notebooks(skill: str, limit: int = KAGGLE_RESULTS_PER_SKILL) -> list:
    try:
        kernels = api.kernels_list(search=skill, sort_by="hotness")
        results = []
        for k in kernels[:limit]:
            results.append({
                "title":  k.title,
                "url":    f"https://www.kaggle.com/code/{k.ref}",
                "author": k.author,
            })
        return results
    except Exception as e:
        print(f"    ⚠️  Notebook search failed for '{skill}': {e}")
        return []



# BUILD RECOMMENDATIONS
def build_recommendations(missing_skills: list) -> list:
    recommendations = []
    for skill in missing_skills:
        print(f"  🔍  Searching Kaggle for: {skill}")
        recommendations.append({
            "skill":     skill,
            "datasets":  fetch_kaggle_datasets(skill),
            "notebooks": [],#fetch_kaggle_notebooks(skill),  
        })
    return recommendations


# PRINT REPORT
def print_report(gaps: dict, recommendations: list) -> None:
    print("\n" + "=" * 60)
    print("📋  GAP SUMMARY")
    print("=" * 60)
    print(gaps["gap_summary"])

    print("\n" + "=" * 60)
    print("📚  KAGGLE RECOMMENDATIONS BY SKILL")
    print("=" * 60)

    for rec in recommendations:
        print(f"\n── {rec['skill'].upper()} ──")

        if rec["datasets"]:
            print("  Datasets:")
            for d in rec["datasets"]:
                size_mb = f"{d['size'] / 1_000_000:.1f} MB" if d["size"] else "unknown"
                print(f"    • {d['title']}  ({d['votes']} votes | {d['downloads']} downloads | {size_mb} | usability: {d['usability']})")
                print(f"      {d['url']}")
                print(f"      Author: {d['author']} | License: {d['license']} | Updated: {d['last_updated']}")
                if d.get("tags"):
                    print(f"      Tags: {', '.join(d['tags'][:5])}")
                if d.get("description"):
                    print(f"      {str(d['description'])[:100]}...")
        else:
            print("  Datasets: (none found)")

        if rec["notebooks"]:
            print("  Notebooks:")
            for n in rec["notebooks"]:
                print(f"    • {n['title']}  by {n['author']}")
                print(f"      {n['url']}")
        else:
            print("  Notebooks: (none found)")


# SAVE FULL REPORT
def save_report(gaps: dict, recommendations: list, output_path: str = REPORT_OUTPUT_FILE) -> None:
    report = {
        "gap_analysis": gaps,
        "kaggle_recommendations": recommendations,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾  Full report saved to: {output_path}")



# FEATURE EXTRACTION (WITH GROUPS)
def extract_features(recommendations: list):

    X = []
    group = []
    dataset_info = []

    for rec in recommendations:
        skill = rec["skill"]
        group_count = 0

        for d in rec["datasets"]:
            votes = d.get("votes", 0) or 0
            downloads = d.get("downloads", 0) or 0
            usability = d.get("usability", 0)

            # skip bad rows
            if usability is None:
                continue

            # features
            log_votes = np.log1p(votes)
            log_downloads = np.log1p(downloads)

            features = [log_votes, log_downloads, usability]

            X.append(features)
            dataset_info.append((skill, d["title"]))  # keep skill for debugging
            group_count += 1

        if group_count > 0:
            group.append(group_count)

    X = np.array(X)

    print("\n📊 Feature Matrix Shape:", X.shape)
    print("📦 Group Sizes:", group)

    return X, group, dataset_info

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    gaps = load_gaps(GAPS_INPUT_FILE)

    print("🚀  Fetching Kaggle resources...")
    recommendations = build_recommendations(gaps["missing_skills"])

    print_report(gaps, recommendations)
    save_report(gaps, recommendations)

    #extract features for xgboost
    X, group, dataset_info = extract_features(recommendations)