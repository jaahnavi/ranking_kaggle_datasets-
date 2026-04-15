# ranking_kaggle_datasets-
Find out which skill you are missing from your resume compared to a job description and find the most suitable datasets you can work on from kaggle

Kaggle Dataset Ranking System for Business Analytics Skill Gaps

This project helps business analytics students bridge skill gaps efficiently by recommending the most relevant Kaggle datasets based on missing skills identified from job descriptions.

Instead of manually searching through hundreds of datasets, this system:

-Identifies missing skills from resume vs job description
-Fetches high-quality Kaggle datasets for each skill
-Uses a learning-to-rank model (XGBoost) to rank datasets
-Recommends top datasets optimized for resume impact

Outcome:
Students focus on high-impact projects that improve job readiness and increase interview chances.

How It Works
1. Skill Gap Extraction
Extracts missing skills from resume vs job description
Implemented in: Resume_gap_extractor.py

2. Kaggle Data Pipeline
Searches Kaggle datasets for each missing skill
Extracts structured metadata (votes, downloads, usability, etc.)
Builds feature matrix + groups for ranking
Implemented in: kaggle_search.py

3. Feature Engineering

Each dataset is represented using:

log(votes)
log(downloads)
usability score

4. Learning-to-Rank Model
Model: XGBoost Ranking (pairwise)
Learns which datasets are better within each skill group
Trained using custom relevance labels

Training script: train_ranker.py

5. Inference (Ranking)
Loads trained model
Scores new datasets
Returns top 3 datasets per skill

Inference script: rank_datasets.py

📂 Project Structure
.
├── Resume_gap_extractor.py   # Extract missing skills
├── kaggle_search.py          # Fetch datasets + feature extraction
├── train_ranker.py           # Train ranking model
├── rank_datasets.py          # Use trained model to rank datasets
├── xgb_ranker.json           # Saved XGBoost model
├── scaler.pkl                # Saved feature scaler
├── gaps.json                 # Input: missing skills
└── kaggle_report.json        # Output: raw dataset results

⚙️ Installation
pip install kaggle xgboost scikit-learn numpy joblib

Also configure Kaggle API:

~/.kaggle/kaggle.json

How to Use

Step 1: Extract Skill Gaps
python Resume_gap_extractor.py --resume resume.pdf --jd job_description.txt --out gaps.json

Step 2: Train the Model
python train_ranker.py

This will:
Fetch datasets
Extract features
Train ranking model
Save:
xgb_ranker.json
scaler.pkl

Step 3: Rank Datasets (Inference)
python rank_datasets.py

This will:
Load trained model
Fetch new datasets
Rank them per skill
Output: top 3 datasets per skill

🧠 Model Details
Algorithm: XGBoost Ranking (rank:pairwise)
Groups: Each skill = one ranking group
Labels:
2 → strong resume project
1 → moderate
0 → weak
🔑 Key Design Decisions
✅ Group-Based Ranking

Datasets are ranked within each skill, not globally.

✅ Feature Scaling

Uses StandardScaler to normalize:

votes
downloads
usability

Saved in:

scaler.pkl
✅ Model Persistence

Saved model:

xgb_ranker.json

Allows:

Reuse without retraining
Modular deployment
🚀 Future Improvements
Add NLP features (title + description embeddings)
Incorporate dataset tags (business relevance)
Improve label generation (automated weak supervision)
Build API (FastAPI / Flask)
Add evaluation metrics (NDCG)
📌 Summary

This project builds a production-style ML ranking system that:

Aligns datasets with career skill gaps
Prioritizes high-impact projects
Uses learning-to-rank instead of simple scoring

👉 Result: smarter, faster, and more strategic project selection for business analytics students.
