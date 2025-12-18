# Model Card

## Model Details
This project trains a binary classification model to predict income class (`>50K` vs `<=50K`) using the Census (Adult) dataset.  
I used a **Logistic Regression** model from scikit-learn. For preprocessing, I one-hot encoded the categorical columns and kept the numeric columns as-is.  
Saved files: `model/model.pkl`, `model/encoder.pkl`, and `model/lb.pkl`.

## Intended Use
This model is for this course project only. The goal is to show a full pipeline (train → evaluate → slice metrics → API).  
It is **not** meant to be used for real decisions about people.

## Training Data
Training data comes from `data/census.csv` (Adult/Census Income dataset).  
I split the data into train/test using an **80/20 split**, stratified by the `salary` label.

## Evaluation Data
Evaluation is done on the 20% test split from the same dataset (`data/census.csv`).

## Metrics
I used **Precision**, **Recall**, and **F1** (F-beta with beta=1).

Test set performance:
- Precision: **0.6940**
- Recall: **0.5886**
- F1: **0.6370**

I also computed slice performance for every unique value of each categorical feature, and saved it in `slice_output.txt`.

## Ethical Considerations
This dataset has sensitive fields like race and sex, so bias can show up in the predictions.  
Because of that, this model should not be used for any high-stakes decision-making.

## Caveats and Recommendations
This is a baseline model for the assignment, not a production model.  
Some slice groups may have small counts, so their metrics can jump around.  
Also, logistic regression may show a convergence warning sometimes, but the pipeline still trains and makes predictions for this project.
