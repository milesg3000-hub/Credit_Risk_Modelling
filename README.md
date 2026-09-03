# Credit_Risk_Modelling
## 1. Overview
   This project investigates the impact of different features on whether Taiwanese customers default on their loan payment, using the UCI dataset that includes 30,000 observations. The features of the dataset are detailed below.
   ### Information about features of UCI dataset

<details>
<summary><strong> Features overview (click to expand)</strong></summary>
<p style="font-size:18px;"><u>There are 25 variables in total with the following descriptions:</u></p>

ID: ID of each client

LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit)

SEX: Gender (1=male, 2=female)

EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)

MARRIAGE: Marital status (1=married, 2=single, 3=others)

AGE: Age in years

PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)

PAY_2: Repayment status in August, 2005 (scale same as above)

PAY_3: Repayment status in July, 2005 (scale same as above)

PAY_4: Repayment status in June, 2005 (scale same as above)

PAY_5: Repayment status in May, 2005 (scale same as above)

PAY_6: Repayment status in April, 2005 (scale same as above)

BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)

BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)

BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)

BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)

BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)

BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)

PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)

PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)

PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)

PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)

PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)

PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)

default.payment.next.month: Default payment (1=yes, 0=no)

</details>

A feature of bill_amt_mean was also created taking the mean of the amount of bill statement across the six months. This decision on feature selection was taken to avoid introducing multicolinearity by including the bill amounts across all of the six months.
The machine learning algorithms applied in this project included logistic regression fitted through MLE and optimized via Newton's method using only Numpy and Pandas, a 4-feature degree-2 map (using features of Repayment Status (Sep 2005), the mean of the bill_amt, age, and credit limit), cost-complexity pruned classification tree, and a custom Random Forest ensemble method built using OOP. Stratified k-cross fold validation (as outlined in the notebook) was used to resolve the train/test split paradigm, testing on the untouched test dataset once whilst keeping a validation set (subset of training data) before the full training set (training + validation) was tested on the test set (out of sample dataset) for each of the algorithms. There were various key metrics computed for all of the algorithms beyond simple ROC-AUC analysis to account for the class imbalance prevalent in the credit risk dataset. Indeed, only approximately 22% of the customers defaulted on their loan payment in the UCI dataset. The metrics included the Gini coefficient, KS statistic and PR-AUC metrics, the details of which are outlined in the notebook.

## 2. Model Performance Comparison
| Model Architecture | Evaluation Set | ROC-AUC | PR-AUC | Gini | K-S Stat | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 5-Fold CV (Val Mean) | 0.7394 | 0.5110 | 0.4787 | 38.35% | 81.86% |
| **Degree 2 Feature Map** | 5-Fold CV (Val Mean) | 0.7447 | 0.5100 | 0.4894 | **38.52%** | 81.86% |
| **Decision Tree (Gini, $\alpha = 0.0002$)** | Holdout Test Set | 0.7501 | 0.4949 | 0.5002 | 37.96% | 81.82% |
| **Decision Tree (Entropy, $\alpha = 0.0007$)** | Holdout Test Set | 0.7528 | **0.5376** | 0.5056 | 37.53% | **81.93%** |
| **Custom Random Forest (OOP)** | Holdout Test Set | **0.7587** | 0.5189 | **0.5174** | 37.98% | 81.62% |
### 3. Interpretation of Results

*   **Model Discrimination & Rank-Ordering (ROC-AUC & Gini):** The **Custom Random Forest (OOP)** architecture demonstrated the highest discriminatory power, achieving a **ROC-AUC of 0.7587** and a **Gini coefficient of 0.5174** on the untouched holdout test set. This confirms that introducing non-linear ensemble methods significantly improves the pipeline's capability to rank-order risk and cleanly separate creditworthy borrowers from potential defaults compared to the baseline logit models.

*   **Mitigating Class Imbalance (PR-AUC vs. Accuracy):** Given the structural class imbalance inherent to this portfolio (where defaults comprise only 22% of observations), standard classification accuracy metrics are a misleading indicator of performance, plateauing uniformly around 81.6%–81.9%. Instead, optimizing the **Decision Tree via cost-complexity pruning ($\alpha = 0.0007$ with Entropy splitting)** yielded the strongest **PR-AUC of 0.5376**. Prioritizing the Precision-Recall curve ensures the pipeline maintains high precision while capturing true default events, directly limiting institutional exposure to catastrophic False Negatives.

*   **Risk Population Separation (K-S Statistic):** The **Degree 2 Feature Map** achieved the highest Kolmogorov-Smirnov statistic at **38.52%** during Stratified 5-Fold Cross-Validation. A K-S statistic approaching the 40% threshold indicates that the cumulative distribution functions of the default and non-default populations are highly differentiated. This provides an optimal mathematical framework for establishing precise credit-scoring cut-off thresholds for risk segmentation.


