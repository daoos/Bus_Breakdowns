import pickle
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def gen_text(pred, true, details):
    if (pred == 1) & (true == 1):
        print(str(details) + " Real")
    elif pred == 1:
        print(str(details) + "False Positive")


def actual_breakdowns(ypred, ytrue, identifiers):
    [gen_text(x, y, z) for x, y, z in zip(ypred, ytrue, identifiers)]
    exit()


def successcalculator(ypred, ytrue):
    yptmp = list(map(int, ypred))
    yttmp = list(map(int, ytrue))
    predictionTypes = []
    for i in range(2):
        for j in range(2):
            predictionTypes.append(sum([(x[0] == i) and (x[1] == j) for x in zip(yttmp, yptmp)]))
    print('True Negatives: ' + str(predictionTypes[0]).ljust(6) + ' | False Positives: ' + str(predictionTypes[1]))
    print('False Negatives: ' + str(predictionTypes[2]).ljust(5) + ' | True Positives: ' + str(predictionTypes[3]))

cutoff_date_1 = "2016-08-01"
cutoff_date_2 = "2016-10-01"
df = pickle.load(open('output/final_df.pkl', 'rb')).fillna(0)
trainSet, testSet = df[df['date'] < pd.Timestamp(cutoff_date_1)], \
                    df[df['date'] >= pd.Timestamp(cutoff_date_2)]
YTeach = trainSet.as_matrix()[:, :3]
XTeach = trainSet.as_matrix()[:, 3:]
YTrue = testSet.as_matrix()[:, :3]
XTrue = testSet.as_matrix()[:, 3:]
YTeach, YTrue, XTeach, XTrue = sklearn.model_selection.train_test_split(df.as_matrix()[:, :3], df.as_matrix()[:, 3:], test_size=.3)
clf = RandomForestClassifier(n_estimators=14, criterion='gini', max_features='log2', max_depth=20, n_jobs=-1, class_weight={0.0: 1, 1.0: 3.5})
clf.fit(XTeach, YTeach[:, 0].astype(np.float32))
YPred = clf.predict(XTrue)
successcalculator(YPred, YTrue[:, 0])
actual_breakdowns(YPred, YTrue[:, 0], YTrue[:, 1:])
pickle.dump(clf, open('output/' + 'Model.pkl', 'wb'))
pickle.dump(list(df.columns), open('output/' + 'column.pkl', 'wb'))
pickle.dump(df, open('output/df.pkl', 'wb'))
