import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import sys

columns = ["Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Serum Cholesterol", "Fasting Blood Sugar", 
           "Resting ECG Results", "Max Heart Rate", "Exercise Induced Angina", "Old Peak", "Slope Peak", 
           "Number of Vessels", "Thalassemia", "Disease"]


df = pd.read_csv('processed.cleveland.data', dtype=object, header=None, names=columns)

features = columns.copy()
features.pop()

defaults = {
    "Sex" : 0,
    "Chest Pain Type" : 1, 
    "Fasting Blood Sugar" : 0,
    "Resting ECG Results" : 0,
    "Exercise Induced Angina" : 0,
    "Thalassemia" : 3,
    "Number of Vessels" : 0,
    "Slope Peak" : 1,
    "Age" : 25,
    "Resting Blood Pressure" : 100,
    "Serum Cholesterol" : 200,
    "Max Heart Rate" : 100,
    "Old Peak" : 2, 
}

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

continuousIncrements = {
    "Age" : [1, 10, 50],
    "Resting Blood Pressure" : [1, 10, 50],
    "Serum Cholesterol" : [1, 10, 50],
    "Max Heart Rate" : [1, 10, 50],
    "Old Peak" : [0.1, 0.5, 2], 
}

continuousLimits = {
    "Age" : [1, 100],
    "Resting Blood Pressure" : [0, 250],
    "Serum Cholesterol" : [0, 600],
    "Max Heart Rate" : [0, 250],
    "Old Peak" : [0, 10], 
}

# used to remember how much I scaled a variable down
scaledBy = {}


# --------------------- normalisation etc ---------------------

# just replace ? with most common value and convert to floats
for col in df.columns:
    mode = df[col].value_counts().index[0]
    df[col] = df[col].replace("?", mode)
    df[col] = df[col].astype(float)

# normalise continuous columns:
for col in continuousIncrements:
    scaledBy[col] = df[col].max()
    df[col] /= df[col].max()

# split up each categorical collumn into multiple columns of 1 or 0
for col in categoricalDict:

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


# --------------------- Neural Network training ---------------------

trainSize = int(0.7*len(df.index))

(x_train, y_train) = (df.drop(['Disease'], axis=1)[:trainSize], df["Disease"][:trainSize])
(x_test,  y_test)  = (df.drop(['Disease'], axis=1)[trainSize:], df["Disease"][trainSize:])

def buildNetwork():

    # input and a gaussian layer as a kind of data augmentation method
    model_input = tf.keras.Input(shape=(len(df.columns)-1,))
    noise = tf.keras.layers.GaussianNoise(stddev=0.2)(model_input)

    # a few dense layers
    d1 = tf.keras.layers.Dense(16, activation='relu')(noise)
    d1_drop = tf.keras.layers.Dropout(0.2)(d1)

    d2 = tf.keras.layers.Dense(16, activation='relu')(d1_drop)
    d2_drop = tf.keras.layers.Dropout(0.2)(d1)

    # residual layers
    d3    = tf.keras.layers.Dense(8, activation='relu')(d2_drop)
    res_1 = tf.keras.layers.Dense(8, activation='relu')(d3)
    add_1 = tf.keras.layers.add([res_1, d3])

    d4    = tf.keras.layers.Dense(8, activation='relu')(add_1)
    res_2 = tf.keras.layers.Dense(8, activation='relu')(d4)
    add_2 = tf.keras.layers.add([res_2, d4])

    d5    = tf.keras.layers.Dense(4, activation='relu')(add_2)
    res_3 = tf.keras.layers.Dense(4, activation='relu')(d5)
    add_3 = tf.keras.layers.add([res_3, d5])

    d6    = tf.keras.layers.Dense(4, activation='relu')(add_3)
    res_4 = tf.keras.layers.Dense(4, activation='relu')(d6)
    add_4 = tf.keras.layers.add([res_4, d5])

    d7    = tf.keras.layers.Dense(4, activation='relu')(add_4)
    res_5 = tf.keras.layers.Dense(4, activation='relu')(d7)
    add_5 = tf.keras.layers.add([res_5, d5])

    # output
    output = tf.keras.layers.Dense(2, activation='softmax')(add_5)

    model = tf.keras.Model(model_input, output)

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model


# use this to predict the outcome for given values, as well as 
# the comparative prediction for all other variables changed
def predictWithComparisons(valuesDict):

    model = tf.keras.models.load_model('model0.h5')

    # build the base value first
    base = [ 0.0 for feature in x_train.columns.values ]
    for feature in valuesDict:
        if feature in categoricalDict:
            # find the name of the column
            category = feature + ": " + categoricalDict[feature][int(valuesDict[feature])]
            # then find where that column is, and set the correspinding value to 1
            location = x_train.columns.values.tolist().index(category)
            base[location] = 1.0
        
        else:
            base[x_train.columns.values.tolist().index(feature)] = float(valuesDict[feature]) / scaledBy[feature]


    # calculate the base prediction
    prediction = model.predict(np.reshape(base, (1, -1)), batch_size=None)[0][1]

    # build up all other combos including the dict to retrun for formatting
    subClasses = {}
    for feature in features:
        subClasses[feature] = []

        # go through all categories
        if feature in categoricalDict:
            oldCat = feature + ": " + categoricalDict[feature][int(valuesDict[feature])]
            oldLoc = x_train.columns.values.tolist().index(oldCat)
            for category in categoricalDict[feature]:
                subClass = {'value': category, 'name': categoricalDict[feature][category]}

                # already have a value set to this
                if int(valuesDict[feature]) == category:
                    subClass['chosen'] = True
                else:
                    subClass['chosen'] = False

                # calc new theoretical score
                newCat = feature + ": " + categoricalDict[feature][category]
                newLoc = x_train.columns.values.tolist().index(newCat)

                temp = base.copy()
                temp[oldLoc] = 0.0
                temp[newLoc] = 1.0

                newPred = model.predict(np.reshape(temp, (1, -1)), batch_size=None)[0][1]

                if prediction > newPred:
                    subClass['fill']   = 'rgba(54, 194, 200, 0.6)'
                    subClass['border'] = 'rgba(54, 194, 200, 1)'
                else:
                    subClass['fill']   = 'rgba(223, 105, 26, 0.6)'
                    subClass['border'] = 'rgba(223, 105, 26, 1)'

                subClasses[feature].append(subClass)
        
        # for each continuous, give + and - increments
        if feature in continuousIncrements:

            value = 0
            if feature == "Old Peak":
                value = float(valuesDict[feature])
            else:
                value = int(valuesDict[feature])

            # negative:
            for inc in reversed(continuousIncrements[feature]):
                val = max(value - inc, continuousLimits[feature][0])
                subClasses[feature].append({'value': val, 'name': str(val), 'chosen': False})
            
            # current:
            subClasses[feature].append({'value': value, 'name': str(value), 'chosen': True})
            
            # positive:
            for inc in continuousIncrements[feature]:
                val = min(value + inc, continuousLimits[feature][1])
                subClasses[feature].append({'value': val, 'name': str(val), 'chosen': False})

            for subClass in subClasses[feature]:
                temp = base.copy()
                temp[x_train.columns.values.tolist().index(feature)] = subClass['value'] / scaledBy[feature]

                newPred = model.predict(np.reshape(temp, (1, -1)), batch_size=None)[0][1]

                if prediction > newPred:
                    subClass['fill']   = 'rgba(54, 194, 200, 0.6)'
                    subClass['border'] = 'rgba(54, 194, 200, 1)'
                elif prediction < newPred:
                    subClass['fill']   = 'rgba(223, 105, 26, 0.6)'
                    subClass['border'] = 'rgba(223, 105, 26, 1)'
                else:
                    # not change...
                    subClass['fill']   = 'rgba(200, 200, 200, 0.6)'
                    subClass['border'] = 'rgba(200, 200, 200, 1)'

    return str(prediction), subClasses

# build networks if desired
def trainNetwork():
    
    model = buildNetwork()

    # max of 50 iterations since it starts overfitting at around then
    best_acc = 0
    for iteration in range(50):
        model.fit(x_train, y_train, epochs=2, verbose=0)
        _, train = model.evaluate(x_train, y_train, verbose=0)
        _, test = model.evaluate( x_test,  y_test, verbose=0)
        
        # save the best one
        if test > best_acc:
            best_acc = test
            print("--- saved with test acc: " + str(test))
            model.save('saved.h5')


# if we run this file, we create 3 models automatically
if __name__ == "__main__":

    # trainNetwork()

    # model = tf.keras.models.load_model('model0.h5')
    # _, test = model.evaluate( x_test,  y_test, verbose=0)
    # print("Accuracy: " + str(test))


