# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
train_loc = '/content/drive/MyDrive/CS4622 - Machine Learning/speech-based-classification-layer-10/train.csv'
valid_loc = '/content/drive/MyDrive/CS4622 - Machine Learning/speech-based-classification-layer-10/valid.csv'
test_loc = '/content/drive/MyDrive/CS4622 - Machine Learning/speech-based-classification-layer-10/test.csv'

# %%
import pandas as pd

# %%
# Load the data
train_df = pd.read_csv(train_loc)
valid_df = pd.read_csv(valid_loc)
test_df = pd.read_csv(test_loc)

# %%
train_df.head()

# %%
valid_df.head()

# %%
test_df.head()

# %%
train_df.tail()

# %%
train_df.shape

# %%
nan_counts_label_1 = train_df['label_1'].isna().sum()
nan_counts_label_2 = train_df['label_2'].isna().sum()
nan_counts_label_3 = train_df['label_3'].isna().sum()
nan_counts_label_4 = train_df['label_4'].isna().sum()

# %%
print(f"NaN count for label 1: {nan_counts_label_1}")
print(f"NaN count for label 2: {nan_counts_label_2}")
print(f"NaN count for label 3: {nan_counts_label_3}")
print(f"NaN count for label 4: {nan_counts_label_4}")

# %%
train_df.describe()

# %%
FEATURES = [f'feature_{i}' for i in range(1,769)]

# %% [markdown]
# ## Label 1

# %%
train1 = train_df[FEATURES + ['label_1']].dropna()
valid1 = valid_df[FEATURES + ['label_1']]

# %%
train1.head()

# %%
from sklearn.preprocessing import StandardScaler

# %%
x_train1 = train1.drop('label_1',axis=1)
y_train1 = train1['label_1']

x_valid1 = valid1.drop('label_1',axis=1)
y_valid1 = valid1['label_1']

# %%
scaler = StandardScaler()

# %%
x_train1 = pd.DataFrame(scaler.fit_transform(x_train1),columns=FEATURES)
x_valid1 = pd.DataFrame(scaler.transform(x_valid1),columns=FEATURES)

# %% [markdown]
# ### Random Forest Classifier with Grid Search

# %%
from sklearn import svm

# %%
clf = svm.SVC(kernel='linear')
clf.fit(x_train1,y_train1)

# %%
y_pred1 = clf.predict(x_valid1)

# %%
from sklearn import metrics

def find_accuracy(Y_pred,Y_valid):
  print(f"accuracy_score {metrics.accuracy_score(Y_valid,Y_pred)}")
  print(f"precision_score {metrics.precision_score(Y_valid,Y_pred,average='weighted')}")
  print(f"recall_score {metrics.recall_score(Y_valid,Y_pred,average='weighted')}")

# %%
find_accuracy(y_pred1,y_valid1)

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=10)

rf_classifier.fit(x_train1, y_train1)

# %%
y_pred1_rf = rf_classifier.predict(x_valid1)

# %%
find_accuracy(y_pred1_rf,y_valid1)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

logistic_params = {
    'C': [0.1, 1, 10],
}

logistic_classifier = LogisticRegression()
logistic_random_search = RandomizedSearchCV(logistic_classifier, param_distributions=logistic_params, cv=5, n_iter=10)
logistic_random_search.fit(x_train1, y_train1)
logistic_best_classifier = logistic_random_search.best_estimator_
logistic_best_params = logistic_random_search.best_params_

# %%
logistic_best_params

# %%
logistic_predictions = logistic_best_classifier.predict(x_valid1)

# %%
find_accuracy(logistic_predictions,y_valid1)

# %%
from sklearn.decomposition import PCA

candidate_n_components = [0.99,.95]

# %% [markdown]
# ### PCA

# %%
best_n_components = None
best_explained_variance = 0.0

# %%
for n in candidate_n_components:
    pca = PCA(n_components=n, svd_solver='full')
    X_train_pca = pca.fit_transform(x_train1)
    explained_variance = pca.explained_variance_ratio_
    total_explained_variance = sum(explained_variance)
    if total_explained_variance > best_explained_variance:
        best_explained_variance = total_explained_variance
        best_n_components = n

# %%
print("Best n_components:", best_n_components)
print("Explained Variance with Best n_components:", best_explained_variance)

# %%
pca = PCA(n_components=best_n_components)
X_train_pca = pca.fit_transform(x_train1)
X_test_pca = pca.transform(x_valid1)
classifier = RandomForestClassifier()
classifier.fit(X_train_pca, y_train1)

y_pred_rf = classifier.predict(X_test_pca)

# %%
X_train_pca.shape

# %%
find_accuracy(y_pred_rf,y_valid1)

# %%
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train_pca, y_train1)
svm_y_pred = svm_classifier.predict(X_test_pca)
find_accuracy(svm_y_pred,y_valid1)

# %%
logistic_classifier = LogisticRegression(C=0.1)
logistic_classifier.fit(X_train_pca, y_train1)
logistic_y_pred = logistic_classifier.predict(X_test_pca)
find_accuracy(logistic_y_pred,y_valid1)

# %% [markdown]
# ### Correlation Co-efficient

# %%
import numpy as np

# %%
corr_matrix1 = x_train1.corr()

# %%
upper_triangle1 = corr_matrix1.where(np.triu(np.ones(corr_matrix1.shape), k=1).astype(np.bool))
correlated_features_to_drop1 = [column for column in upper_triangle1.columns if any(upper_triangle1[column] > 0.5)]

# %%
x_train1.shape

# %%
print("Correlated features to drop:", correlated_features_to_drop1)
print("Number of features to drop:", len(correlated_features_to_drop1))
print("Number of features left:", x_train1.shape[1] - len(correlated_features_to_drop1))

# %%
x_train_filtered1 = x_train1.drop(columns=correlated_features_to_drop1)
x_valid_filtered1 = x_valid1.drop(columns=correlated_features_to_drop1)

# %%
print("Number of features left after dropping correlated features:", x_train_filtered1.shape[1])

# %%
from sklearn.model_selection import GridSearchCV

# %%
logistic_params = {
    'C': [0.1, 1, 10],
}

logistic_classifier = LogisticRegression()
logistic_grid_search = GridSearchCV(logistic_classifier, param_grid=logistic_params, cv=5)
logistic_grid_search.fit(x_train_filtered1, y_train1)

logistic_best_classifier = logistic_grid_search.best_estimator_
logistic_best_params = logistic_grid_search.best_params_

# %%
print("logistic_best_classifier",logistic_best_classifier)
print("logistic_best_params",logistic_best_params)

# %%
logistic_predictions = logistic_best_classifier.predict(x_valid_filtered1)

# %%
find_accuracy(logistic_predictions,y_valid1)

# %%
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(x_train_filtered1, y_train1)
svm_y_pred = svm_classifier.predict(x_valid_filtered1)
find_accuracy(svm_y_pred,y_valid1)

# %% [markdown]
# ### LogisticRegression will use for the label1's classification

# %%
test_df.head()

# %%
test_df = pd.read_csv(test_loc)

# %%
test_df = test_df.drop('ID', axis=1)

# %%
test_df1 = pd.DataFrame(scaler.transform(test_df),columns=FEATURES)

# %%
test_df_filtered1 = test_df.drop(columns=correlated_features_to_drop1,axis=1)

# %%
test_df_filtered1.shape

# %%
label1_predictions = logistic_best_classifier.predict(test_df_filtered1)

# %%
l1_predictions_df = pd.DataFrame({'label_1': label1_predictions})

l1_predictions_df.to_csv('label_1.csv', index=False)

# %%
l1_predictions_df.shape

# %% [markdown]
# ## Lable 02

# %%
train2 = train_df[FEATURES + ['label_2']].dropna()
valid2 = valid_df[FEATURES + ['label_2']].dropna()

# %%
train2.head()

# %%
from sklearn.preprocessing import StandardScaler

# %%
x_train2 = train2.drop('label_2',axis=1)
y_train2 = train2['label_2']

x_valid2 = valid2.drop('label_2',axis=1)
y_valid2 = valid2['label_2']

# %%
scaler2 = StandardScaler()

# %%
x_train2 = pd.DataFrame(scaler2.fit_transform(x_train2),columns=FEATURES)
x_valid2 = pd.DataFrame(scaler2.transform(x_valid2),columns=FEATURES)

# %% [markdown]
# ### without feature engineering

# %%
clf = svm.SVC(kernel='linear')
clf.fit(x_train2,y_train2)

# %%
y_pred2 = clf.predict(x_valid2)

# %%
find_accuracy(y_pred2,y_valid2)

# %%
best_n_components_2 = None
best_explained_variance_2 = 0.0

# %%
for n in candidate_n_components:
    pca = PCA(n_components=n, svd_solver='full')
    X_train_pca = pca.fit_transform(x_train2)
    print(f"shape : {X_train_pca.shape[1]} for n = {n}")
    explained_variance = pca.explained_variance_ratio_
    total_explained_variance = sum(explained_variance)
    if total_explained_variance > best_explained_variance_2:
        best_explained_variance_2 = total_explained_variance
        best_n_components_2 = n

# %%
print("Best n_components:", best_n_components_2)
print("Explained Variance with Best n_components:", best_explained_variance_2)

# %%
corr_matrix2 = x_train2.corr()

# %%
upper_triangle2 = corr_matrix2.where(np.triu(np.ones(corr_matrix2.shape), k=1).astype(np.bool))
correlated_features_to_drop2 = [column for column in upper_triangle2.columns if any(upper_triangle2[column] > 0.5)]

# %%
print("Correlated features to drop:", correlated_features_to_drop2)
print("Number of features to drop:", len(correlated_features_to_drop2))
print("Number of features left:", x_train2.shape[1] - len(correlated_features_to_drop2))

# %%
x_train_filtered2 = x_train2.drop(columns=correlated_features_to_drop2)
x_valid_filtered2 = x_valid2.drop(columns=correlated_features_to_drop2)

# %%
x_valid_filtered2.shape[1]

# %%
x_train_filtered2.shape[1]

# %%
logistic_params = {
    'C': [0.1, 1, 10],
}

logistic_classifier_2 = LogisticRegression()
logistic_grid_search_2 = GridSearchCV(logistic_classifier_2, param_grid=logistic_params, cv=5)
logistic_grid_search_2.fit(x_train_filtered2, y_train2)

logistic_best_classifier_2 = logistic_grid_search_2.best_estimator_
logistic_best_params_2 = logistic_grid_search_2.best_params_
print("logistic_best_params",logistic_best_params_2)

# %%
print("logistic_best_params",logistic_best_params_2)

# %%
logistic_predictions_2 = logistic_best_classifier_2.predict(x_valid_filtered2)

# %%
find_accuracy(logistic_predictions_2,y_valid2)

# %%
svm_classifier2 = svm.SVC(kernel='linear')
svm_classifier2.fit(x_train_filtered2, y_train2)
svm_y_pred2 = svm_classifier2.predict(x_valid_filtered2)
find_accuracy(svm_y_pred2,y_valid2)

# %%
pca2 = PCA(n_components=best_n_components_2)
X_train_pca2 = pca2.fit_transform(x_train2)
X_test_pca2 = pca2.transform(x_valid2)
classifier2 = RandomForestClassifier()
classifier2.fit(X_train_pca2, y_train2)

y_pred_rf2 = classifier2.predict(X_test_pca2)

# %%
find_accuracy(y_pred_rf2,y_valid2)

# %%
svm_classifier2 = svm.SVC(kernel='linear')
svm_classifier2.fit(X_train_pca2, y_train2)
svm_y_pred2 = svm_classifier2.predict(X_test_pca2)
find_accuracy(svm_y_pred2,y_valid2)

# %%
logistic_classifier2 = LogisticRegression(C=1)
logistic_classifier2.fit(X_train_pca2, y_train2)
logistic_y_pred2 = logistic_classifier2.predict(X_test_pca2)
find_accuracy(logistic_y_pred2,y_valid2)

# %% [markdown]
# ### PCA & svm will use for the label2's classification

# %%
test_df2 = pd.DataFrame(scaler2.transform(test_df),columns=FEATURES)

# %%
test_df2.head()

# %%
test_df_filtered2 = test_df2.drop(columns=correlated_features_to_drop2,axis=1)

# %%
test_df_filtered2.head()

# %%
label2_predictions = svm_classifier2.predict(pca2.transform(pd.DataFrame(scaler2.transform(test_df),columns=FEATURES)))

# %%
label2_predictions_test = svm_classifier2.predict(pca2.transform(x_train2))

# %%
find_accuracy(label2_predictions_test,y_train2)

# %%
label2_predictions

# %%
l2_predictions_df = pd.DataFrame({'label_2': label2_predictions})

l2_predictions_df.to_csv('label_2.csv', index=False)

# %%
l2_predictions_df.shape

# %% [markdown]
# ## Label 3

# %%
train3 = train_df[FEATURES + ['label_3']].dropna()
valid3 = valid_df[FEATURES + ['label_3']].dropna()

# %%
train3.head()

# %%
from sklearn.preprocessing import StandardScaler

# %%
x_train3 = train3.drop('label_3',axis=1)
y_train3 = train3['label_3']

x_valid3 = valid3.drop('label_3',axis=1)
y_valid3 = valid3['label_3']

# %%
scaler3 = StandardScaler()

# %%
x_train3 = pd.DataFrame(scaler3.fit_transform(x_train3),columns=FEATURES)
x_valid3 = pd.DataFrame(scaler3.transform(x_valid3),columns=FEATURES)

# %% [markdown]
# ### without feature engineering

# %%
clf = svm.SVC(kernel='linear')
clf.fit(x_train3,y_train3)

# %%
y_pred3 = clf.predict(x_valid3)

# %%
find_accuracy(y_pred3,y_valid3)

# %%
best_n_components_3 = None
best_explained_variance_3 = 0.0

# %%
for n in candidate_n_components:
    pca = PCA(n_components=n, svd_solver='full')
    X_train_pca = pca.fit_transform(x_train3)
    print(f"shape : {X_train_pca.shape[1]} for n = {n}")
    explained_variance = pca.explained_variance_ratio_
    total_explained_variance = sum(explained_variance)
    if total_explained_variance > best_explained_variance_3:
        best_explained_variance_3 = total_explained_variance
        best_n_components_3 = n

# %%
print("Best n_components:", best_n_components_3)
print("Explained Variance with Best n_components:", best_explained_variance_3)

# %%
corr_matrix3 = x_train3.corr()

# %%
upper_triangle3 = corr_matrix3.where(np.triu(np.ones(corr_matrix3.shape), k=1).astype(np.bool))
correlated_features_to_drop3 = [column for column in upper_triangle3.columns if any(upper_triangle3[column] > 0.5)]

# %%
print("Correlated features to drop:", correlated_features_to_drop3)
print("Number of features to drop:", len(correlated_features_to_drop3))
print("Number of features left:", x_train3.shape[1] - len(correlated_features_to_drop3))

# %%
x_train_filtered3 = x_train3.drop(columns=correlated_features_to_drop3)
x_valid_filtered3 = x_valid3.drop(columns=correlated_features_to_drop3)

# %%
x_valid_filtered3.shape[1]

# %%
x_train_filtered3.shape[1]

# %%
logistic_params = {
    'C': [0.1, 1, 10],
}

logistic_classifier_3 = LogisticRegression()
logistic_grid_search_3 = GridSearchCV(logistic_classifier_3, param_grid=logistic_params, cv=5)
logistic_grid_search_3.fit(x_train_filtered3, y_train3)

logistic_best_classifier_3 = logistic_grid_search_3.best_estimator_
logistic_best_params_3 = logistic_grid_search_3.best_params_
print("logistic_best_params",logistic_best_params_3)

# %%
print("logistic_best_params",logistic_best_params_3)

# %%
logistic_predictions_3 = logistic_best_classifier_3.predict(x_valid_filtered3)

# %%
find_accuracy(logistic_predictions_3,y_valid3)

# %%
svm_classifier3 = svm.SVC(kernel='linear')
svm_classifier3.fit(x_train_filtered3, y_train3)
svm_y_pred3 = svm_classifier3.predict(x_valid_filtered3)
find_accuracy(svm_y_pred3,y_valid3)

# %%
pca3 = PCA(n_components=best_n_components_3)
X_train_pca3 = pca3.fit_transform(x_train3)
X_test_pca3 = pca3.transform(x_valid3)
classifier3 = RandomForestClassifier()
classifier3.fit(X_train_pca3, y_train3)

y_pred_rf3 = classifier3.predict(X_test_pca3)

# %%
find_accuracy(y_pred_rf3,y_valid3)

# %%
svm_classifier3 = svm.SVC(kernel='linear')
svm_classifier3.fit(X_train_pca3, y_train3)
svm_y_pred3 = svm_classifier3.predict(X_test_pca3)
find_accuracy(svm_y_pred3,y_valid3)

# %%
logistic_classifier3 = LogisticRegression(C=0.1)
logistic_classifier3.fit(X_train_pca3, y_train3)
logistic_y_pred3 = logistic_classifier3.predict(X_test_pca3)
find_accuracy(logistic_y_pred3,y_valid3)

# %% [markdown]
# ### LogisticRegression will use for the label3's classification

# %%
test_df3 = pd.DataFrame(scaler3.transform(test_df),columns=FEATURES)

# %%
test_df3.head()

# %%
test_df_filtered3 = test_df3.drop(columns=correlated_features_to_drop3,axis=1)

# %%
test_df_filtered3.head()

# %%
label3_predictions = logistic_best_classifier_3.predict(test_df_filtered3)

# %%
label3_predictions

# %%
l3_predictions_df = pd.DataFrame({'label_3': label3_predictions})

l3_predictions_df.to_csv('label_3.csv', index=False)

# %%
l3_predictions_df.shape

# %% [markdown]
# ## Label 4

# %%
train4 = train_df[FEATURES + ['label_4']].dropna()
valid4 = valid_df[FEATURES + ['label_4']].dropna()

# %%
train4.head()

# %%
from sklearn.preprocessing import StandardScaler

# %%
x_train4 = train4.drop('label_4',axis=1)
y_train4 = train4['label_4']

x_valid4 = valid4.drop('label_4',axis=1)
y_valid4 = valid4['label_4']

# %%
scaler4 = StandardScaler()

# %%
x_train4 = pd.DataFrame(scaler4.fit_transform(x_train4),columns=FEATURES)
x_valid4 = pd.DataFrame(scaler4.transform(x_valid4),columns=FEATURES)

# %% [markdown]
# ### without feature engineering

# %%
clf = svm.SVC(kernel='linear')
clf.fit(x_train4,y_train4)

# %%
y_pred4 = clf.predict(x_valid4)

# %%
find_accuracy(y_pred4,y_valid4)

# %%
best_n_components_4 = None
best_explained_variance_4 = 0.0

# %%
for n in candidate_n_components:
    pca = PCA(n_components=n, svd_solver='full')
    X_train_pca = pca.fit_transform(x_train4)
    print(f"shape : {X_train_pca.shape[1]} for n = {n}")
    explained_variance = pca.explained_variance_ratio_
    total_explained_variance = sum(explained_variance)
    if total_explained_variance > best_explained_variance_4:
        best_explained_variance_4 = total_explained_variance
        best_n_components_4 = n

# %%
print("Best n_components:", best_n_components_4)
print("Explained Variance with Best n_components:", best_explained_variance_4)

# %%
corr_matrix4 = x_train4.corr()

# %%
upper_triangle4 = corr_matrix4.where(np.triu(np.ones(corr_matrix4.shape), k=1).astype(np.bool))
correlated_features_to_drop4 = [column for column in upper_triangle4.columns if any(upper_triangle4[column] > 0.5)]

# %%
print("Correlated features to drop:", correlated_features_to_drop4)
print("Number of features to drop:", len(correlated_features_to_drop4))
print("Number of features left:", x_train4.shape[1] - len(correlated_features_to_drop4))

# %%
x_train_filtered4 = x_train4.drop(columns=correlated_features_to_drop4)
x_valid_filtered4 = x_valid4.drop(columns=correlated_features_to_drop4)

# %%
x_valid_filtered4.shape[1]

# %%
x_train_filtered4.shape[1]

# %%
logistic_params = {
    'C': [0.1, 1, 10],
}

logistic_classifier_4 = LogisticRegression()
logistic_grid_search_4 = GridSearchCV(logistic_classifier_4, param_grid=logistic_params, cv=5)
logistic_grid_search_4.fit(x_train_filtered4, y_train4)

logistic_best_classifier_4 = logistic_grid_search_4.best_estimator_
logistic_best_params_4 = logistic_grid_search_4.best_params_
print("logistic_best_params",logistic_best_params_4)

# %%
print("logistic_best_params",logistic_best_params_4)

# %%
logistic_predictions_4 = logistic_best_classifier_4.predict(x_valid_filtered4)

# %%
find_accuracy(logistic_predictions_4,y_valid4)

# %%
svm_classifier4 = svm.SVC(kernel='linear')
svm_classifier4.fit(x_train_filtered4, y_train4)
svm_y_pred4 = svm_classifier4.predict(x_valid_filtered4)
find_accuracy(svm_y_pred4,y_valid4)

# %%
pca4 = PCA(n_components=best_n_components_4)
X_train_pca4 = pca4.fit_transform(x_train4)
X_test_pca4 = pca4.transform(x_valid4)
classifier4 = RandomForestClassifier()
classifier4.fit(X_train_pca4, y_train4)

y_pred_rf4 = classifier4.predict(X_test_pca4)

# %%
X_train_pca4.shape

# %%
find_accuracy(y_pred_rf4,y_valid4)

# %%
svm_classifier4 = svm.SVC(kernel='linear')
svm_classifier4.fit(X_train_pca4, y_train4)
svm_y_pred4 = svm_classifier4.predict(X_test_pca4)
find_accuracy(svm_y_pred4,y_valid4)

# %%
logistic_classifier4 = LogisticRegression(C=0.1)
logistic_classifier4.fit(X_train_pca4, y_train4)
logistic_y_pred4 = logistic_classifier4.predict(X_test_pca4)
find_accuracy(logistic_y_pred4,y_valid4)

# %% [markdown]
# ### LogisticRegression will use for the label3's classification

# %%
test_df4 = pd.DataFrame(scaler4.transform(test_df),columns=FEATURES)

# %%
test_df4.head()

# %%
test_df_filtered4 = test_df4.drop(columns=correlated_features_to_drop4,axis=1)

# %%
test_df_filtered4.head()

# %%
label4_predictions = logistic_best_classifier_4.predict(test_df_filtered4)

# %%
label4_predictions

# %%
l4_predictions_df = pd.DataFrame({'label_4': label4_predictions})

l4_predictions_df.to_csv('label_4.csv', index=False)

# %%
l4_predictions_df.shape

# %% [markdown]
# ## Final Output

# %%
df1 = pd.read_csv('/content/label_1.csv')
df2 = pd.read_csv('/content/label_2.csv')
df3 = pd.read_csv('/content/label_3.csv')
df4 = pd.read_csv('/content/label_4.csv')

# %%
df1.shape

# %%
merged_df = pd.concat([df1, df2, df3, df4], axis=1)

# %%
merged_df["ID"] = range(1, len(df1) + 1)

# %%
desired_column_order = ['ID', 'label_1', 'label_2', 'label_3', 'label_4']

# %%
merged_df = merged_df[desired_column_order]

# %%
merged_df.head()

# %%
merged_df.to_csv('190023G_layer10.csv', index=False)


