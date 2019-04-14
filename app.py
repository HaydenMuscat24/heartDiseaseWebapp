from flask import Flask
from flask import Markup
from flask import Flask
from flask import render_template
from flask import request


import tensorflow as tf

import stats
import logits
import neuralNet

app = Flask(__name__)

@app.route("/")
def dataVisualisationPage():
    feature = request.args.get('feature') or "Sex"
    rate = request.args.get('rate') or "totals"
    labels = stats.getLabels(feature)
    data = stats.getData(feature, rate)
    return render_template('visualisation.html', feature=feature, rate=rate, labels=labels, features=stats.features,
                           dataWith=data["with"], dataWithout=data["without"])

@app.route("/importance")
def featureImportancePage():

    # any updates to the feature selection?
    selected = request.args.get('selected') or ''
    remove  = request.args.get('remove')
    add     = request.args.get('add')
    accuracy = "--"

    # maybe we start a list?
    if selected == '' and add != None:
        selected = add 
    
    # split if it has elements
    if selected != '':
        selected = selected.split(",")
        if remove != None and remove in selected:
            selected.remove(remove)
        if add != None and add not in selected:
            selected.append(add)
        print(selected)
        if len(selected) > 0:
            accuracy = str(logits.runLRwith(selected))[:4]
    
        selected = ','.join(selected)

    # get the base correlation
    coefficients, labels, fillColors, borderColors = logits.getFeatureCoefficients()
    return render_template('importance.html', selected=selected, labels=labels, features=logits.features, 
                           coefficients=coefficients, fillColors=fillColors, borderColors=borderColors, 
                           accuracy=accuracy)

@app.route("/prediction")
def predictionPage():
    
    # get the chosen values
    values = {}
    for feature in neuralNet.features:
        values[feature] = request.args.get(feature) or neuralNet.defaults[feature]
    
    # get prediction of values, as well as a return dictionary of related classes for formatting
    prediction, subClasses = neuralNet.predictWithComparisons(values)

    return render_template('prediction.html', features=neuralNet.features, catDict=neuralNet.categoricalDict,
                           prediction=prediction, subClasses=subClasses)
 
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
