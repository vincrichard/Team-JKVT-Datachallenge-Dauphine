# [Data Challenge Dauphine](https://www.qminitiative.org/hackathon2---intelligence-artificielle-&-machine-learning.html) 

Our Team JKVT finished third of the challenge it was compose of four people:

 * Jérémie Peres, [jeremieperes](https://github.com/jeremieperes)
 * Kévin Ferin, 
 * Vincent Richard, [vincrichard](https://github.com/vincrichard)
 * Thomas Rivière [t-riviere](https://github.com/t-riviere) 

## Goal of the Data Challenge

**Subject**: Volatility prediction on earnings announcement dates

**Goal**: The purpose of this topic is to predict the volatile behaviour of a stock on the day of its earnings release.

**The Metric Evaluated**: AUC

**Result** 

<img src=result.png></img>

## Sum Up of the Approach

### Phase 1: 6pm - 11am

This part was a first approach to the problem and the testing of different preprocessing techniques and models. As we had no knowledge of the financial field, it was difficult for us to create new relevant variables.

We measured the quality of our models by a score of no means on a 3 folds cross validation, which allowed us not to overfiter the test set.

Preprocessing :

* Drop the raw_id column
* One Hot Encoding of categorical variables ie sectors and currency
* Processing missing values by replacing them with a constant value -9999
* Variables are modified by non-linear functions, but these modifications have not been kept.

The models tested:

* Random Forest
* Neural Network
* Different linear models for stacking purposes
* Light GBM
* XGBoost

In parallel, we tested another approach by training two different LGBM models:

* one based on Stoxx 600 stock data.
* one on SP500 stock data

Our best results were obtained with an XGBoost on the whole data following the search for hyperparameters with an improved Random Search via the Hyperopt library which uses the Tree-structured Parzen Estimator (TPE) algorithm. This algorithm intelligently explores the hyperparameter search space by progressively limiting itself to the best estimated hyperparameters.

### Phase 2: 2pm-7pm

On this second phase of the hackathon, we decided **to focus on the preprocessing** of the feature matrix before model training. Our idea was to code all the preprocessing ideas we had before testing them during our model training to see if they had a positive impact on our scores.

To do this we did the following pre-processing:

  *  Drop the raw_id column
  * One Hot Encoding of categorical variables ie sectors and currency
  * Processing missing values by choosing one or more of the following techniques:
  * Creating variables indicating whether the value is missing for ebit and ebitda
  * Fill in the missing values with :
        
    - a KNN
    - the sector average of the variable
    - the mean of the variable
    - the median of the variable
    - the most frequent value for the variable
    - a constant value : -9999, -999, 0, 999 or 9999.
* Creation of features: weekly / monthly / yearly difference for the variables return and realized volatility
* Over-sampling via the [SMOTE](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html) technique in order to obtain as many examples of each class for training purposes.
* Normalization/Standardization of variables with a MinMaxScaler or StandardScaler

We then focused on the **XGBoost and LightGBM models**.

We did a **hyperparameter search with an improved Random Search via the [Hyperopt](https://github.com/hyperopt/hyperopt)** library which uses the Tree-structured Parzen Estimator (TPE) algorithm. This algorithm intelligently explores the hyperparameter search space by progressively limiting itself to the best estimated hyperparameters.

This allows us to test different hyperparameters for our models but also for our preprocessing.

For the selection of the best model during this hyperparameter search, we performed a **10 folds cross-validation** and we created a **customized score** in order to take into account both the mean of the cross-validation scores and their variance:

```python
if scoring_std>=0.1:
        scoring = 0
    else:
        scoring = scoring_mean / max(0.01, scoring_std)
```

Finally, we have re-trained the best model on the completeness of the data.

Unfortunately, none of the models in this second phase was a priori better than the one we had in the first phase. The best models we had had a better average score on all the cross-validation folds but the standard deviation of these scores was too large for us to take the risk of submitting a new model when we were first in the first phase. We therefore made the prudent choice to submit the same model as in phase 1.

We also wanted to try to stack several different models (Logistic Regression, SVM with Gaussian kernel, KNN, Random Forest, Gradient Boosting, Neural Networks...) but we didn't have the time to do so.


