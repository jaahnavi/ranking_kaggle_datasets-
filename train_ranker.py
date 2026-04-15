import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler

# import your existing script
from kaggle_search import load_gaps, build_recommendations, extract_features


# ─────────────────────────────────────────────
# LOAD DATA + EXTRACT FEATURES
# ─────────────────────────────────────────────
gaps = load_gaps("gaps.json")
recommendations = build_recommendations(gaps["missing_skills"])

X, group, dataset_info = extract_features(recommendations)


# ─────────────────────────────────────────────
# NORMALIZE FEATURES
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n After normalization:")
print(X_scaled[:3])


# ─────────────────────────────────────────────
#Labels
# ─────────────────────────────────────────────

y_by_skill = {
    "Classification": [2, 2, 2, 2, 2, 1, 0, 0, 0, 0],
    "Regression": [2, 2, 1, 1, 1, 1, 0, 1, 0, 0],
    "Time Series Forecasting": [2, 2, 2, 2, 1, 1, 1, 1, 1, 0],
    "NLP": [2, 2, 2, 1, 2, 1, 1, 0, 0, 0],
    "A/B Testing": [2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
}

skill_order = [
    "Classification",
    "Regression",
    "Time Series Forecasting",
    "NLP",
    "A/B Testing"
]

y = []
for skill in skill_order:
    y.extend(y_by_skill[skill])

y = np.array(y)

print("\nSample labels:", y[:10])


# ─────────────────────────────────────────────
# BUILD XGBOOST RANKING MODEL
# ─────────────────────────────────────────────
dtrain = xgb.DMatrix(X_scaled, label=y)
dtrain.set_group(group)

params = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg",
    "max_depth": 4,
    "eta": 0.1,
    "verbosity": 1
}

model = xgb.train(params, dtrain, num_boost_round=50)

print("\n Model trained successfully!")


# ─────────────────────────────────────────────
# PREDICT (RANKING)
# ─────────────────────────────────────────────
scores = model.predict(dtrain)

# attach scores to dataset info
results = list(zip(dataset_info, scores))

# Sort within each group

start = 0

print("\n🏆 Top 3 datasets per skill:\n")

for g in group:
    end = start + g

    # slice this group
    group_items = dataset_info[start:end]
    group_scores = scores[start:end]

    # zip + sort
    group_pairs = list(zip(group_items, group_scores))
    group_sorted = sorted(group_pairs, key=lambda x: x[1], reverse=True)

    # get skill name
    skill = group_sorted[0][0][0]

    print(f"\n=== {skill.upper()} ===")

    # print top 3
    for i in range(min(3, len(group_sorted))):
        ((skill, title), score) = group_sorted[i]
        print(f"{i+1}. {title}  → {score:.4f}")

    start = end


# ─────────────────────────────────────────────
# SAVE MODEL + SCALER
# ─────────────────────────────────────────────
model.save_model("xgb_ranker.json")

joblib.dump(scaler, "scaler.pkl")

print("\n💾 Model and scaler saved!")
