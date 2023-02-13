import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('/Users/.../.../filename.csv')
features = tpot_data.drop('Churn', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Churn'], random_state=42)

# Average CV score on the training set was: 0.9874709639953542
exported_pipeline = make_pipeline(
    PCA(iterated_power=9, svd_solver="randomized"),
    KNeighborsClassifier(n_neighbors=3, p=1, weights="distance")
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

#Adding a line to output the resutls
print(results)
