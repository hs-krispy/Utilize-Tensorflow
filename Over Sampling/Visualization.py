import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SMOTENC, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
data = load_breast_cancer()
forest = RandomForestClassifier(random_state=42)
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = pd.DataFrame(data.target, columns=["target"])
index = df[df['target'] == 0][:162].index.tolist()
x = df.drop(index=index)
y = x['target']
x = x.drop(columns=["target"])
print(y.value_counts())
y = np.ravel(y)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
# forest.fit(X_train, y_train)
# print("Original set\n{}".format(classification_report(y_test, forest.predict(X_test))))
pca = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
X_vis = pca.fit_transform(X_train)

# Apply the random over-sampling
ada = KMeansSMOTE(random_state=42)
X_resampled, y_resampled = ada.fit_sample(X_train, y_train)
y_resampled = np.ravel(y_resampled)
forest.fit(X_resampled, y_resampled)
print(Counter(y_resampled))
print(y_resampled.shape)
X_res_vis = pca.transform(X_resampled)

print("KMeansSMOTE\n{}".format(classification_report(y_test, forest.predict(X_test))))


f, (ax1, ax2) = plt.subplots(1, 2)

c0 = ax1.scatter(X_vis[y_train == 0, 0], X_vis[y_train == 0, 1], label="Class #0", alpha=0.5)
c1 = ax1.scatter(X_vis[y_train == 1, 0], X_vis[y_train == 1, 1], label="Class #1", alpha=0.5)
ax1.set_title('Original set')

ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1], label="Class #0", alpha=.5)
ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1], label="Class #1", alpha=.5)
ax2.set_title('KMeansSMOTE')

# make nice plotting
for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # ax.set_xlim([-700, 1700])
    # ax.set_ylim([-200, 200])

plt.figlegend((c0, c1), ('Class #0', 'Class #1'), loc='lower center', ncol=2, labelspacing=0.)
plt.tight_layout(pad=3)
plt.show()