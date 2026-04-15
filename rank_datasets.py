import numpy as np
import xgboost as xgb
import joblib

# import your existing functions
from kaggle_search import load_gaps, build_recommendations, extract_features


# ─────────────────────────────────────────────
# LOAD MODEL + SCALER
# ─────────────────────────────────────────────
model = xgb.Booster()
model.load_model("xgb_ranker.json")

scaler = joblib.load("scaler.pkl")

print("✅ Model loaded")


# ─────────────────────────────────────────────
# GET NEW DATA
# ─────────────────────────────────────────────
gaps = load_gaps("gaps.json")
recommendations = build_recommendations(gaps["missing_skills"])

X, group, dataset_info = extract_features(recommendations)


# ─────────────────────────────────────────────
# APPLY SAME NORMALIZATION
# ─────────────────────────────────────────────
X_scaled = scaler.transform(X)


# ─────────────────────────────────────────────
# PREDICT SCORES
# ─────────────────────────────────────────────
dtest = xgb.DMatrix(X_scaled)
scores = model.predict(dtest)


# ─────────────────────────────────────────────
# RANK WITHIN EACH GROUP
# ─────────────────────────────────────────────
start = 0

print("\n🏆 Top 3 datasets per skill:\n")

for g in group:
    end = start + g

    group_items = dataset_info[start:end]
    group_scores = scores[start:end]

    group_pairs = list(zip(group_items, group_scores))
    group_sorted = sorted(group_pairs, key=lambda x: x[1], reverse=True)

    skill = group_sorted[0][0][0]

    print(f"\n=== {skill.upper()} ===")

    for i in range(min(3, len(group_sorted))):
        ((skill, title), score) = group_sorted[i]
        print(f"{i+1}. {title} → {score:.4f}")

    start = end