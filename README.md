# ML Assignment 2

## a. Problem statement
This project addresses a direct marketing problem from a Portuguese banking institution. The historical campaign data captures phone-based marketing interactions with clients, where in many cases multiple contacts were required before a client made a final decision.

The classification goal is to predict whether a client will subscribe to a term deposit, represented by the target variable `y`:
1. `yes` -> client subscribed
2. `no` -> client did not subscribe

From a business perspective, this is a response prediction task that helps prioritize outreach, improve campaign efficiency, and reduce unnecessary follow-up calls by identifying clients with a higher probability of subscription.

This repository implements the full ML workflow for that problem:
1. Data validation, preprocessing, and EDA for campaign data.
2. Training multiple classification models on the same dataset.
3. Comparing models using required evaluation metrics.
4. Building a Streamlit app where users upload test CSV data, select a model, and view performance outputs.
5. Preparing the solution for Streamlit Community Cloud deployment.

## b. Dataset description
Dataset source: UCI Bank Marketing dataset
- URL: https://archive.ics.uci.edu/dataset/222/bank+marketing

Dataset used in this repository:
1. Training file: train_data.csv
2. Test file for app evaluation: test_data.csv
3. Delimiter: semicolon (`;`)

High-value dataset notes:
1. Data parsing: CSV is semicolon-separated and loaded with `sep=";"` in both training and Streamlit evaluation flow.
2. Feature design: 20 input features are grouped into customer profile, campaign/contact history, and macroeconomic indicators.
3. Class imbalance: target distribution is no=29276 (88.85%) and yes=3674 (11.15%), approximately an 8:1 ratio.
4. Reproducibility protocol: stratified 80/20 split with random_state=42, and preprocessing is fit on training data only before transforming evaluation/test data.
5. Metric rationale: because of class imbalance, model comparison uses AUC, Recall, F1, and MCC in addition to Accuracy.

Dataset properties (from train_data.csv):
1. Instances (rows): 32950
2. Input features: 20
3. Target column: y
4. Classification type: Binary (yes/no)

Attribute details:
1. Numeric attributes:
	1. age
	2. duration
	3. campaign
	4. pdays
	5. previous
	6. emp.var.rate
	7. cons.price.idx
	8. cons.conf.idx
	9. euribor3m
	10. nr.employed
2. Categorical attributes:
	1. job
	2. marital
	3. education
	4. default
	5. housing
	6. loan
	7. contact
	8. month
	9. day_of_week
	10. poutcome
3. Target attribute:
	1. y (`yes`/`no`)

Target distribution in training data:
1. no: 29276
2. yes: 3674

Preprocessing and validation applied:
1. Missing value checks
2. Duplicate row checks and removal
3. Outlier analysis (IQR-based summary)
4. Train-fit and test-transform preprocessing with ColumnTransformer

EDA visuals generated:
1. Target distribution
2. Subscription rate by job

## c. Github Repository and Streamlit App Link
- GitHub Repo: https://github.com/2025da04199/2025DA04199_ML_Assignment2

- Streamlit App: https://2025da04199-ml-assignment2.streamlit.app/

## d. Models used and evaluation metrics
Implemented models:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble)

Evaluation metrics used for each model:
1. Accuracy
2. AUC
3. Precision
4. Recall
5. F1
6. MCC

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8636 | 0.9341 | 0.4434 | 0.8735 | 0.5882 | 0.5594 |
| Decision Tree | 0.8953 | 0.7245 | 0.5323 | 0.5048 | 0.5182 | 0.4597 |
| KNN | 0.9018 | 0.8563 | 0.5866 | 0.4054 | 0.4795 | 0.4362 |
| Naive Bayes | 0.8144 | 0.8289 | 0.3298 | 0.6435 | 0.4361 | 0.3657 |
| Random Forest (Ensemble) | 0.9044 | 0.9421 | 0.5558 | 0.7116 | 0.6241 | 0.5760 |

### Model performance observations

| ML Model Name | Observation about Model Performance |
| --- | --- |
| Logistic Regression | Logistic Regression obtained an accuracy of 86.36% and an AUC of 0.9341, which shows that it has strong overall classification capabilities and is very good at distinguishing between the two classes. It reached the highest recall of 87.35%, showing that it was very effective in identifying the actual positive cases. Yet its precision was relatively low at 44.34%, this indicating that it had a high number of false-positive predictions. Overall, Logistic Regression is most appropriate when the main aim is to maximise recall and minimise the number of missed positive cases. |
| Decision Tree | The Decision Tree had an accuracy of 89.53 per cent, which shows that its overall ability to carry out classification is good. Nevertheless, its AUC of 0.7245 was much lower than that of the other models which performed well, implying that it had weaker capacity to discriminate between the classes. Although its precision of 53.23 per cent and recall of 50.48 per cent were fairly balanced though only moderate. With an F1-score of 0.5182 and an MCC of 0.4597, the Decision Tree showed a reasonable but comparatively weaker overall performance. |
| KNN | KNN attained an accuracy of 90.18 per cent and thus came in second place when the models were ranked by accuracy. It had the highest precision at 58.66 per cent, which shows that its positive predictions were fairly reliable. Yet its recall was only 40.54 per cent, the lowest of all the models, indicating that it failed to detect a large number of the actual positive cases. As a result, its F1-score of 0.4795 and its MCC of 0.4362 were both relatively low, showing that its high accuracy did not lead to good overall classification performance. |
| Naive Bayes | The Naive Bayes model had the lowest accuracy at 81.44 percent of all the models examined. Even though its AUC score was 0.8289 and its recall amounted to 64.35 per cent, showing that it has a reasonable capacity to distinguish and identify positive cases, its precision was only 32.98 per cent, the lowest of all the models. This indicates that the model produced a relatively large number of false positive predictions. Furthermore, its F1-score of 0.4361 and its MCC of 0.3657 were also the lowest, so Naive Bayes was the worst-performing model in this comparison. |
| Random Forest (Ensemble) - Overall Winner | Random Forest obtained the highest accuracy at 90.44% and the highest AUC of 0.9421, which shows that it has very good overall classification performance and a strong ability to distinguish between classes. The method achieved a precision of 55.58% and a recall of 71.16%, meaning that it offered a much better balance between correctly identifying positive cases and keeping the number of false positives low. Its F1-score of 0.6241 was the highest of all the models, proving that it had the best balance between precision and recall. Moreover, it had the highest MCC at 0.5760, thus confirming its high level of overall predictive reliability. For these reasons, Random Forest is awarded the title of Overall Winner since it delivered the best and most consistent results across most of the evaluation criteria. |

## Project structure

project-folder/
|- streamlit_app.py
|- requirements.txt
|- README.md
|- train_data.csv
|- test_data.csv
|- model/
|  |- train_models.py
|  |- preprocess.py
|  |- evaluation_utils.py
|  |- eda.py
|  |- config/
|  |- output/
|  |- *.joblib

## Python version
Use Python 3.10 for this project.

## How to run

1. Create and activate virtual environment:

macOS/Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Train models and generate artifacts + outputs:

```bash
python model/train_models.py
```

4. Run Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Streamlit features implemented
1. CSV upload for test data.
2. Model selection dropdown.
3. Evaluation metrics display.
4. Confusion matrix and classification report display.

