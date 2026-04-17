# Enterprise Telecom Churn Prediction: A Value-Driven ML Pipeline
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-1793D1?style=for-the-badge&logo=xgboost&logoColor=white) ![CUDA](https://img.shields.io/badge/NVIDIA_GPU_Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

> **Objective:** Bridging the gap between raw mathematical accuracy and real-world business retention strategy. 

## ⬢ Executive Summary
Customer churn costs enterprise telecommunications companies millions annually. Standard machine learning models often optimize for global *accuracy*, completely missing the minority class (the actual churning customers). 

This project discards the baseline approach. By implementing synthetic data generation (SMOTE), hardware-accelerated ensemble learning (XGBoost via CUDA), and strategic probability threshold calibration, this pipeline successfully intercepts **96% of churning customers**—translating predictive analytics into direct revenue retention.

## ⬢ Technical Architecture
This is a full-stack, end-to-end data pipeline optimized for a local NVIDIA GPU environment:

1. **Data Ingestion & Formatting:** - Handled hidden data corruption (e.g., coerced whitespace errors in continuous numerical columns) and enforced strict `float64` and `Int64` typing.
   - Dropped non-predictive cardinality (Customer IDs) to prevent model overfitting.
2. **Feature Engineering:**
   - Translated human business context into algorithm-ready mathematics via robust One-Hot Encoding (`pd.get_dummies`) and Boolean to Integer conversion.
   - Applied `StandardScaler` to ensure uniform loss-landscape navigation during gradient descent.
3. **Class Imbalance Resolution:**
   - Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) strictly on the `X_train` matrix, mathematically generating realistic minority vectors to force the algorithm to learn churn patterns rather than memorizing class biases.
4. **Hardware-Accelerated Modeling (XGBoost):**
   - Bypassed CPU bottlenecks by routing the XGBoost `hist` tree method directly to a dedicated NVIDIA GPU (`device='cuda'`). 
   - Deployed `GridSearchCV` with 3-fold cross-validation to isolate the optimal hyperparameter architecture (Max Depth: 3, Learning Rate: 0.01) specifically targeting the `Recall` metric.

## ⬢ Business Value & Impact
A model is only as valuable as the money it saves. By aggressively lowering the prediction threshold from the default `0.50` to a customized `0.35`, the model was tuned to reflect the reality of B2C marketing: sending a $10 retention discount to a false positive is infinitely cheaper than losing a $1,000 customer entirely.

| Metric | Baseline Model (Logistic Regression) | Tuned Architecture (XGBoost + SMOTE) | Business Impact |
| :--- | :--- | :--- | :--- |
| **Recall (Churn)** | 60% | **96%** | **Intercepts nearly 100% of flight-risk accounts.** |
| **Precision (Churn)** | 68% | **38%** | Widens the net. Accepts controlled false alarms to guarantee maximum retention. |

## ⬢ Local Deployment
To replicate this pipeline locally, ensure your environment is configured for GPU acceleration. 

```bash
# Clone the repository
git clone [https://github.com/jemmziray-tech/enterprise-churn-interception.git](https://github.com/jemmziray-tech/enterprise-churn-interception.git)
```
# Install strictly defined dependencies
```bash
pip install -r requirements.txt
```
# Note: Ensure local NVIDIA drivers are optimized for CUDA acceleration to utilize the XGBoost GPU tree method.

Engineered by John Elifuraha Mziray — BSc Artificial Intelligence & Machine Learning, University of Limerick.
