# shodh_AI_assignment_Policy-Optimization_Decision-Making



This repository contains all steps (EDA, Deep Learning model, and Reinforcement Learning agent and evaluation matrix comparison) implemented in a single Jupyter Notebook using Google Colab.

##  Repository Content
Policy_Optimization_for_Financial_Decision_Making_.ipynb – Contains the complete workflow (EDA → DL model → RL agent->evaluation matrix-> comparision->decision making )

##  Setup Instructions

### Run in Google Colab (Recommended)
1. Open the notebook in Google Colab by clicking the badge below:  

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balujaldu/shodh_AI_assignment_Policy-Optimization_Decision-Making/blob/main/Policy_Optimization_for_Financial_Decision_Making_.ipynb)

2. Run all cells (`Runtime > Run all`).

## Dataset Access
This project uses a custom dataset stored in Google Drive.  
You can access it here: [Dataset Link](https://drive.google.com/file/d/19glLMndsUNvN2KLC0x23cwwibWiTOfD1/view?usp=sharing)

⚠️ After opening in Colab:
- Mount Google Drive (`from google.colab import drive; drive.mount('/content/drive')`)
- Update the `data_path` variable in the first cell of the notebook to point to your dataset path.

if it doesnt work you can access my colab book with the below link : 

## Open in Google Colab
[Click here to view the notebook](https://colab.research.google.com/drive/1J8vH1jdn84isUNe1YgGlKeb9TTOGCPQf?usp=sharing)

# step-by-step explanation of my environment and code for this problem statement :

# Policy Optimization for Financial Decision-Making

Comparative Analysis between Multi-Layer Perceptron and Reinforcement Learning Agent

This project focuses on building a policy decision-making system for loan approval by comparing Supervised Deep Learning (MLP) and Reinforcement Learning (RL) agents. The pipeline involves data preprocessing, model training, evaluation with standard metrics, and critical case analysis where models may disagree.

 # Project Workflow
 
# Task 1: Exploratory Data Analysis (EDA) and Preprocessing

Performed on Lending Club loan dataset to prepare clean, balanced, and normalized data.

Step 1–2: Imported dataset from Google Drive, loaded CSV file, and performed basic visualization.

Step 3: Inspected datatypes of each column.

Step 4: Applied feature engineering by removing irrelevant columns with no impact on target prediction.

Step 5–6: Re-checked datatypes and used .describe() to summarize numeric features.

Step 7–9: Printed unique values, counted missing values, and removed rows with null entries.

Step 10–11: Checked class imbalance and converted target column (loan_status) into binary:

1 = Fully Paid

0 = Others (Charged Off, Late, etc.)

Step 12: Dropped the original loan_status column.

Step 13: Converted categorical variables into numeric using one-hot encoding (get_dummies).

Step 14: Removed outliers using the standard deviation method to ensure fairer learning.

Step 15: Verified balance of target classes.

Step 16: Normalized numerical features for consistent scaling across variables.

# Task 2: Multi-Layer Perceptron (MLP) Implementation

Step 17: Imported libraries required for deep learning model implementation.

Step 18: Trained an MLP classifier with 3 hidden layers, using K-Fold Cross Validation to handle imbalance.

Step 19: Since performance was limited, applied SMOTE (Synthetic Minority Oversampling) to balance the dataset.

Step 20: Further balanced the dataset by selectively removing excess "Fully Paid" rows to improve fairness and accuracy.

Step 21: Evaluated model using F1-score and ROC-AUC, plotted the ROC Curve to measure performance.

# Task 3: Offline Reinforcement Learning Agent (RL)

Step 22: Designed a custom RL environment (LoanApprovalEnv) using OpenAI Gym:

State (s): Feature vector of an applicant.

Actions (a): {0 = Deny Loan, 1 = Approve Loan}.

Reward (r):

Deny → 0 (neutral)

Approve + Fully Paid → + loan_amount × interest_rate (profit)

Approve + Default → - loan_amount (loss)

Each applicant is treated as a one-step episode.

Step 23: Split dataset and passed loan_amnt and int_rate into the environment. Installed required packages.

Step 24: Wrapped the environment with Stable Baselines3 (SB3) and trained a Deep Q-Learning (DQN) agent.

Step 25: Evaluated the RL agent using policy evaluation (mean reward ± std) over 1000 simulated applicants.

# Task 4: Comparative Analysis & Critical Case Study

Step 26: Compared MLP vs. RL Agent performance using:

F1-score

ROC-AUC

Estimated Policy Value (mean reward)

Step 27: Performed critical applicant analysis to identify cases where MLP and RL policies disagree:

Example: MLP rejects a loan due to high predicted default probability, but RL agent approves because expected profit (interest × loan amount) outweighs risk.

This step highlights how reward-driven RL differs from risk-minimizing supervised learning.





# Evaluation Metrics

MLP Model: F1-score & ROC-AUC.
RL Agent: Estimated Policy Value (Mean Reward ± Std).
Comparison: Identified disagreements on critical applicants to analyze policy trade-offs.


# Future Work

Tune RL rewards with more realistic profit-loss structures.
Experiment with advanced offline RL algorithms (e.g., CQL, BCQ).
Integrate explainability methods (e.g., SHAP) for applicant-level decisions.

# conclusion
This project demonstrates how policy optimization for financial decision-making can be approached using both supervised deep learning and reinforcement learning, providing insights into accuracy vs. reward trade-offs.


## 📄 Final Report
Please see the attached `Final_Report.pdf` for detailed methodology, results, and answers to Task 4.

## 📄 Resume
My updated Resume/CV is attached in the git repo and submission email.



