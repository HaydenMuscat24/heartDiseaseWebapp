import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression

columns = ["Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Serum Cholesterol", "Fasting Blood Sugar", 
           "Resting ECG Results", "Max Heart Rate", "Exercise Induced Angina", "Old Peak", "Slope Peak", 
           "Number of Vessels", "Thalassemia", "Disease"]

df = pd.read_csv('processed.cleveland.data', dtype=object, header=None, names=columns)

features = columns
features.pop()

categoricalDict= {
    "Sex" : {0: "Female", 1: "Male"},
    "Chest Pain Type" : {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}, 
    "Fasting Blood Sugar" : {0: "Above 120 mg/dl", 1: "Below 120 mg/dl"},
    "Resting ECG Results" : {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"},
    "Exercise Induced Angina" : {0: "No", 1: "Yes"},
    "Thalassemia" : {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"},
    "Number of Vessels" : {0: "0", 1: "1", 2: "2", 3: "3"},
    "Slope Peak" : {1: "1", 2: "2", 3: "3"}
}

continuousDict = {
    "Age" : [0, 10, 80],
    "Resting Blood Pressure" : [50, 10, 200],
    "Serum Cholesterol" : [100, 50, 600],
    "Max Heart Rate" : [60, 20, 220],
    "Old Peak" : [0, 0.5, 6.5], 
}

# --------------------- normalisation etc ---------------------

# just replace ? with most common value and convert to floats
for col in df.columns:
    mode = df[col].value_counts().index[0]
    df[col] = df[col].replace("?", mode)
    df[col] = df[col].astype(float)

# normalise continuous columns:
for col in continuousDict:
    df[col] /= df[col].max()

# split up each categorical collumn into multiple columns of 1 or 0
for col in categoricalDict:

    # only split up a category if it has more than 2 categories
    # if int(df[col].max()) <= 1:
    #     continue
    
    # make the new columns
    for key in categoricalDict[col]:
        newName = col + ": " + categoricalDict[col][key]
        df[newName] = 0.0

    # set row value depending on initial value
    for i in range(len(df.index)):
        category = col + ": " + categoricalDict[col][int(df[col][i])]
        df[category][i] = 1.0

    # drop the initial col
    df = df.drop([col], axis=1)


# change disease collumn to 1 if not 0, otherwise 0
df.loc[df['Disease'] != 0.0, 'Disease'] = 1.0


# --------------------- Logistic Regression of whole thing ---------------------

def getFeatureCoefficients():

    results = []

    lr = LogisticRegression(solver='liblinear')
    (x, y) = (df.drop(['Disease'], axis=1), df["Disease"])
    lr.fit(x, y)

    results = list(zip(lr.coef_[0], df.drop(['Disease'], axis=1).columns))

    def sortScore(res): 
        return abs(res[0]) 

    results.sort(key = sortScore, reverse=True)  
    
    coefficients = []
    borderColors = []
    fillColors = []
    labels = []

    pos = "rgba(223, 105, 26, "
    neg = "rgba(54, 194, 200, "

    for x in results:
        coefficients.append(abs(x[0]))
        labels.append(x[1])
        if x[0] > 0:
            fillColors.append(pos+"0.6)")
            borderColors.append(pos+"1)")
        else:
            fillColors.append(neg+"0.6)")
            borderColors.append(neg+"1)")

    return coefficients, labels, fillColors, borderColors

# --------------------- For given features? ---------------------

def runLRwith(selected):

    trainSize = int(0.7*len(df.index))

    new = selected.copy()

    for feature in selected:
        if feature in categoricalDict:
            for category in categoricalDict[feature]:
                new.append(feature + ": " + categoricalDict[feature][category])
            new.remove(feature)

    lr = LogisticRegression(solver='liblinear')

    (x_train, y_train) = (df[new][:trainSize], df["Disease"][:trainSize])
    (x_test,  y_test)  = (df[new][trainSize:], df["Disease"][trainSize:])

    lr.fit(x_train, y_train)

    return lr.score(x_test, y_test)
    