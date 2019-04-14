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

def getLabels(feature):

    labels = []

    # if categorigal, build it up from the dictionary
    if feature in categoricalDict:
        for key in categoricalDict[feature]:
            labels.append(categoricalDict[feature][key])

    # for continuous ones, use the bins as decided
    elif feature in continuousDict:

        (start, step, end) = continuousDict[feature]
        lows = list(np.arange(start, end + step, step)) 
        highs = list(np.arange(start + step, end + 2 * step, step)) 
        labels = [str(x) + " to " + str(y) for (x, y) in zip(lows, highs)]
    
    else:
        print("Error: " + feature + " not found.")
    
    return labels

def getData(feature, rate):

    data = {"with": [], "without": []}

    # if catgegorical I want counts
    if feature in categoricalDict:

        countsWith    = df[df["Disease"] != '0'][feature].value_counts().to_dict()
        countsWithout = df[df["Disease"] == '0'][feature].value_counts().to_dict()

        for key in categoricalDict[feature]:
            key = str(float(key))
            if key not in countsWithout:
                countsWithout[key] = 0
            if key not in countsWith:
                countsWith[key] = 0

            data["with"].append(countsWith[key])
            data["without"].append(countsWithout[key])
    
    # for continuous ones, use the bins as decided
    elif feature in continuousDict:

        (start, step, end) = continuousDict[feature]
        ranges = list(np.arange(start, end + step, step))

        countsWith    = df[(df["Disease"] != '0') & (df[feature] != '?')][feature].astype(float)
        countsWithout = df[(df["Disease"] == '0') & (df[feature] != '?')][feature].astype(float)

        data["with"]    = list(countsWith.groupby(   pd.cut(countsWith,    ranges)).count().to_dict().values())
        data["without"] = list(countsWithout.groupby(pd.cut(countsWithout, ranges)).count().to_dict().values())

    else:
        print("Error: " + feature + " not found.")

    # if its percentage base, we want to divide into percentages
    if rate == "percentages":
        for i in range(len(data["with"])):
            total = data["with"][i] + data["without"][i]
            if total > 0:
                data["with"][i] *= 100/total
                data["without"][i] *= 100/total

    return data
