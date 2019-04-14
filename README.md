# heartDiseaseWebapp

## Running the Webapp
With Flask, Tensorflow, Pandas (all for python 3+) installed, run:

`export FLASK_APP=app.py`
`python -m flask run`

And open up a browser to http://127.0.0.1:5000/

## Creating a Different Neural Network
Run python neuralNet.py, which will automatically run a network on a training set of the data for 50 iterations and save the best iteration as tested on a test set. The file is saved as __saved.h5__, wheras the NN file used within the webapp is __network0.h5__, so just overwrite the latter with the former for use within the app. 

The network architecture is composed of 2 fully connected layers of 16 neurons, then 3 residual layers of width 4, before a softmax (since we're working with binary data). For regularisation, the FC layers have dropout, and for data augmentation and further regularisation, the input goes straight into a gaussian noise layer.

Nontheless, the models begin to overfit at around the 40th iteration (the dataset is tiny, at 300, too small for a NN application), so I keep a max iteration of 50 when training.