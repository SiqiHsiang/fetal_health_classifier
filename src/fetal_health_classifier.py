#!/usr/bin/env python
# coding: utf-8

# ## Fetal Health Classifier

# 
# 
# 
# This project tackles the fetal health classification problem using random forest, XGBoost and various ensemble learning methods, for better performances. 
# 
# This notebook mainly focuses on:
# - preprocess and oversample imbalanced datasets.
# - use random forest, Adaboost, and XGBoost.
# - use grid search (recap on DAMI) and randomized search.
# - use the scikit-learn pipeline to construct a complete and reusable preprocessing cycle.
# - eventually integrate grid search and pipeline to create an automated evaluation process to find out the best ensemble model for dataset
# 
# Dataset:
# 
# This project uses the [fetal health classification](https://www.kaggle.com/andrewmvd/fetal-health-classification) data from Kaggle. The reduction of child mortality is reflected in several of the United Nations' Sustainable Development Goals and is a crucial indicator of human progress. The UN expects that by 2030, countries end preventable deaths of newborns and children under five years of age, with all countries aiming to reduce under‑5 mortality to at least as low as 25 per 1,000 live births.
# 
# Parallel to the notion of child mortality is of course maternal mortality, which accounts for 295 000 deaths during and following pregnancy and childbirth (as of 2017). The vast majority of these deaths (94%) occurred in low-resource settings, and most could have been prevented.
# 
# In light of what was mentioned above, Cardiotocograms (CTGs) are a simple and cost-accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality. The equipment itself works by sending ultrasound pulses and reading its response, thus shedding light on fetal heart rate (FHR), fetal movements, uterine contractions, and more.
# 
# This dataset contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by three expert obstetricians into three classes:
# 
# - Normal: 1
# - Suspect: 2
# - Pathological: 3

# ### 1. Data Exploration

# In[1]:


import pandas as pd
import numpy as np
import imblearn 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv("fetal_health.csv")


# In[3]:


data.head()


# In[4]:


# Check for duplicates in data
data.duplicated()


# In[5]:


# How many duplicates
data.duplicated().sum()


# In[6]:


# drop_duplicates
data_dropped = data.drop_duplicates()


# In[7]:


# Check dropped datapoints and compare to before dropping duplicates
data_dropped


# In[8]:


data


# In[9]:


X = data_dropped.iloc[:, :-1]
y = data_dropped["fetal_health"].astype(int) 


# In[10]:


X.head()


# In[11]:


y.head()


# In[12]:


X.describe()


# In[13]:


import seaborn as sns


# In[14]:


# countplot - check the proportion visually
sns.countplot(x=y)


# In[15]:


# series.value counts - check the proportion
y.value_counts()


# In[16]:


# proportion
y.value_counts() / len(y)


# In[17]:


# Missing data check
X.isna()


# In[18]:


X.isna().sum()


# In[19]:


X.duplicated().sum()


# In[20]:


X_dropped = X[~X.duplicated()]
y_dropped = y[~X.duplicated()]


# In[21]:


# Correlation check - among the features
mat = X_dropped.corr()
mask = np.triu(np.ones_like(mat, dtype=bool))
fig, ax = plt.subplots(figsize = (15,10))
ax = sns.heatmap(mat, mask=mask, annot = True, linewidths=0.5, fmt = ".2f", square=True)


# In[22]:


# Correlation check - features and the label
plt.figure(figsize=(5, 12))
heatmap = sns.heatmap(data.corr()[['fetal_health']].sort_values(by='fetal_health', ascending=True), fmt="0.2f", annot=True)
heatmap.set_title('Features correlating with the label')
heatmap.set_ylim([0,22])


# ### 2. Preprocessing

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X_dropped, y_dropped, test_size=0.3, stratify=y_dropped)


# **Use Scikit-learn to Standardization**

# In[25]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X=X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
X_test_scaled = scaler.transform(X=X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)


# In[26]:


# standardized mean and standard deviation
X_train_scaled.describe()


# In[27]:


y_train.value_counts()


# **Use imblearn to Oversampling for the minority classes(2 and 3)**

# In[28]:


from imblearn.over_sampling import RandomOverSampler


# In[29]:


# Define RandomOverSampler to a variable ros
# ratio
ros = RandomOverSampler(random_state=12345) 
# Create X_resampled, y_resampled using the fit_resample method or fit + sample.
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)


# In[30]:


# value_counts
y_train_resampled.value_counts()


# Unfortunately, this simple oversampler just duplicates the instances we already have. So I put more weights on the minority classes.

# In[31]:


fig, ax = plt.subplots(1, 2, figsize=[8, 4])
ax[0].scatter(X_train_resampled.iloc[:, 0], X_train_resampled.iloc[:, 7], c=y_train_resampled, s=20, linewidth=0.5, edgecolor='black')
ax[1].scatter(X_train_scaled.iloc[:, 0], X_train_scaled.iloc[:, 7], c=y_train, s=20, linewidth=0.5, edgecolor='black')
ax[0].set_title("Resampled")
ax[1].set_title("Original")
plt.show()


# **Use SMOTE or ADASYN to Resample**

# In[32]:


from imblearn.over_sampling import SMOTE, ADASYN


# In[33]:


X_resampled_SMOTE, y_resampled_SMOTE = SMOTE().fit_resample(X_train_scaled, y_train)

X_resampled_ADASYN, y_resampled_ADASYN = ADASYN().fit_resample(X_train_scaled, y_train)


# In[34]:


fig, ax = plt.subplots(1, 4, figsize=[16, 4])
ax[0].scatter(X_train_resampled.iloc[:, 0], X_train_resampled.iloc[:, 7], c=y_train_resampled, s=20, linewidth=0.5, edgecolor='black')
ax[1].scatter(X_resampled_SMOTE.iloc[:, 0], X_resampled_SMOTE.iloc[:, 7], c=y_resampled_SMOTE, s=20, linewidth=0.5, edgecolor='black')
ax[2].scatter(X_resampled_ADASYN.iloc[:, 0], X_resampled_ADASYN.iloc[:, 7], c=y_resampled_ADASYN, s=20, linewidth=0.5, edgecolor='black')
ax[3].scatter(X_train_scaled.iloc[:, 0], X_train_scaled.iloc[:, 7], c=y_train, s=20, linewidth=0.5, edgecolor='black')
ax[0].set_title("Resampled_RANDOM")
ax[1].set_title("Resampled_SMOTE")
ax[2].set_title("Resampled_ADASYN")
ax[3].set_title("Original")

plt.show()


# We keep the minority classes (2 and 3) at most 50% of the majority(class 1).

# ### 3. Simple Ensemble

# 
# **Use VotingClassifier to do a simple ensemble model with a hard/soft voting**

# In[35]:


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[36]:


dt = DecisionTreeClassifier()
lr = LogisticRegression()
svm = SVC()


# In[37]:


voting = VotingClassifier([('lr', lr), ('dt', dt), ('svm', svm)], voting='hard')
voting


# In[38]:


# fit the method on X_train_scaled and y_train
voting.fit(X_train_scaled, y_train)


# Check the scores of each method and the voting classifier.

# In[39]:


for clf in (dt, lr, svm, voting):
  clf.fit(X_resampled_SMOTE, y_resampled_SMOTE)
  print(clf.__class__.__name__, clf.score(X_test_scaled, y_test))


# In[40]:


for clf in (dt, lr, svm, voting):
  clf.fit(X_resampled_ADASYN, y_resampled_ADASYN)
  print(clf.__class__.__name__, clf.score(X_test_scaled, y_test))


# In[41]:


for clf in (dt, lr, svm, voting):
  clf.fit(X_train_scaled, y_train)
  print(clf.__class__.__name__, clf.score(X_test_scaled, y_test))


# In[42]:


from sklearn.metrics import classification_report


# performance scores:

# In[43]:


y_pred = voting.predict(X_test_scaled)
print(classification_report(y_test, y_pred))


# ### 4. Try More Ensemble Methods

# #### Bagging

# In[44]:


from sklearn.ensemble import BaggingClassifier


# In[45]:


# max_features, bootstrap_features
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500, bootstrap = True, n_jobs = -1)
bag_clf.fit(X_train_scaled, y_train)
bag_clf.score(X_test_scaled, y_test)


# Out-of-bag evaluation: Average the scores using the out of bag samples.

# In[46]:


bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500, bootstrap = True, n_jobs = -1, oob_score=True)
bag_clf.fit(X_train_scaled, y_train)
bag_clf.score(X_test_scaled, y_test)


# In[47]:


bag_clf.oob_score_


# In[48]:


bag_clf.oob_decision_function_


# #### Random forests = Decision tree trained via bagging

# In[49]:


from sklearn.ensemble import RandomForestClassifier


# In[50]:


rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)
rf.score(X_test_scaled, y_test)


# #### AdaBoost

# In[51]:


from sklearn.ensemble import AdaBoostClassifier


# In[52]:


ab = AdaBoostClassifier()
ab.fit(X_train_scaled, y_train)
print(ab.score(X_test_scaled, y_test))


# In[53]:


ab = AdaBoostClassifier(estimator=LogisticRegression())
ab.fit(X_train_scaled, y_train)
print(ab.score(X_test_scaled, y_test))


# #### XGBoost

# In[54]:


import xgboost as xgb


# In[55]:


from sklearn.preprocessing import LabelEncoder

# Encode the labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Then train the model
xgbm = xgb.XGBClassifier()
xgbm.fit(X_train_scaled, y_train_encoded)
print(xgbm.score(X_test_scaled, y_test_encoded))


# ### 5. Grid search and Randomized search

# #### Grid Search

# In[56]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[57]:


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# #### Randomized Search

# In[58]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, n_jobs = -1)


# In[64]:


get_ipython().run_cell_magic('time', '', 'rf_random.fit(X_train_scaled, y_train)\n')


# In[65]:


# best_estimator
rf_random.best_estimator_


# In[66]:


# best_score
rf_random.best_score_


# In[67]:


# test score
rf_random.score(X_test_scaled, y_test)


# In[68]:


gs_random = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 3, verbose=2, n_jobs = -1)


# In[69]:


get_ipython().run_cell_magic('time', '', 'gs_random.fit(X_train_scaled, y_train)\n')


# ### 6. Pipelines

# In[70]:


from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures


# In[71]:


pipe = Pipeline([
  ('featureGenerator', PolynomialFeatures()),
  ('scaler', StandardScaler()),
  ('selector', VarianceThreshold(0.1)),
  ('classifier', RandomForestClassifier())
])


# In[72]:


pipe.fit(X_train_scaled, y_train)


# In[73]:


print('Training set score: ' + str(pipe.score(X_train_scaled, y_train)))
print('Test set score: ' + str(pipe.score(X_test_scaled, y_test)))


# #### Add private functions to pipelines

# In[74]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# In[75]:


class SquaredFeatureTransformer(BaseEstimator, TransformerMixin):
  #This class will transform the dataset by adding squared features
  #You need to change the transform function to square each feature's values and add them to the matrix.

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_squared = X ** 2
        return np.hstack((X, X_squared))

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('squared', SquaredFeatureTransformer()),
    ('selector', VarianceThreshold(threshold=0.1)),
    ('classifier', RandomForestClassifier())
])


# #### Integrate gird search into pipelines

# In[76]:


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier

# Define the parameter grid
parameters = {
    'scaler': [StandardScaler(), MinMaxScaler()],
    'classifier': [AdaBoostClassifier(), RandomForestClassifier()],
    'selector__threshold': [0, 0.001, 0.01]
}

# Run grid search
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=parameters,
    cv=5
)


# #### Nested k-fold cross Validation

# In[77]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

data = load_breast_cancer()
X, y = data.data, data.target

# Define the parameter grid
parameters = {
    'RandomForest': {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10]
    },
    'AdaBoost': {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.01, 0.1]
    }
}

# Run grid search (in nested CV)
final_accuracy = {}
outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

for model_name in parameters:
    scores = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', VarianceThreshold()),
            ('classifier',
             RandomForestClassifier(random_state=42) if model_name == 'RandomForest'
             else AdaBoostClassifier(random_state=42))
        ])

        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=pipeline,
                                   param_grid=parameters[model_name],
                                   cv=inner_cv)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = np.mean(y_pred == y_test)
        scores.append(acc)

    final_accuracy[model_name] = np.mean(scores)


# In[82]:


final_accuracy


# ### 7. Test and Visualized Result

# In[ ]:


import ipytest, pytest
ipytest.autoconfig()


# In[96]:


import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

@pytest.fixture
def sample_data():
    # Create a small synthetic dataset similar to fetal_health.csv
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.choice([1, 2, 3], size=100), name='fetal_health')
    return X, y

def test_pipeline_fit_predict(sample_data):
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(0.1)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    assert len(y_pred) == len(y_test)
    # Check that predictions are in the expected set
    assert set(np.unique(y_pred)).issubset({1, 2, 3})

def test_classification_report_and_confusion_matrix(sample_data):
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(0.1)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    assert '1' in report and '2' in report and '3' in report
    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.close()  

def test_export_predictions_to_csv(tmp_path, sample_data):
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(0.1)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    result_df = X_test.copy()
    result_df['true_label'] = y_test.values
    result_df['predicted_label'] = y_pred
    out_csv = tmp_path / "classification_results.csv"
    result_df.to_csv(out_csv, index=False)
    # Check file exists and has correct columns
    loaded = pd.read_csv(out_csv)
    assert 'true_label' in loaded.columns
    assert 'predicted_label' in loaded.columns
    assert len(loaded) == len(y_test)


# In[97]:


ipytest.run('-q') 


# In[98]:


print(classification_report(y_test, y_pred))


# In[106]:


import pandas as pd

# make result table
results = pd.DataFrame({
    "true_label": y_test,
    "predicted_label": y_pred
})

# mark whether prediction was correct
results["correct"] = results["true_label"] == results["predicted_label"]


# In[108]:


results.to_csv("classification_test_results.csv", index=False)


# In[109]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.scatter(range(len(y_test)), y_test, label="True", marker='o')
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", marker='x')
plt.legend()
plt.xlabel("Sample index")
plt.ylabel("Class label")
plt.title("True vs Predicted labels")
plt.show()

