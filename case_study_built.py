# Implementing a Self Organizing Map + ANN 

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values 

# Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM 

# Importing pre-built som implementation 
from minisom import MiniSom 
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
# Assigning random weights of SOM to start with
som.random_weights_init(X)
# Randomly training SOM
som.train_random( data = X, num_iteration = 100)

# Visualizing the Results

# Importing Libraries
from pylab import bone, pcolor, colorbar, plot, show

# Initializes window 
bone()

# Stores all Mean interneuron distance (MID) values
pcolor(som.distance_map().T)

# Will give legend to determine what color is fraudulent (which would be when the value is closest to one)
colorbar()

# Will identify whether or not the customers had an accepted application or not
markers = ['o', 's']
colors = ['r', 'g']

# For loop where each node is centrally marked with either a red circle for failing the application or a 
# green square for passing the application. i represents the index of customers and x represents all of a 
# given customer's information. Loop ensures location, type of marker, outside coolor of marker, inside 
# color of marker, size of marker, size of marker edge, in that order. 
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, 
         markers[y[i]], 
         markeredgecolor = colors[y[i]], 
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the identity of the fraudulent actors
mappings = som.win_map(X)

# Making a list of all frauds with scaled data
frauds = np.concatenate((mappings[(8,3)], mappings[(3,7)]), axis = 0)

# Transforming the data back into its original form to read the ids of the people
frauds = sc.inverse_transform(frauds)


# Part 2: Going from unsupervised to supervised deep learning

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values 

# Creating the dependant variable in the form of a vector storing all the fradulent people as 1s
# and the innocent people as 0s
is_fraud = np.zeros(len(dataset)) 
for i in range (len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
        

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2: Making the ANN

# Importing the Keras libraries and packages
from keras.models import Sequential 
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer 
classifier.add(Dense(units = 2, kernel_initializer = 'uniform' , activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(output_dim = 1, kernel_initializer  = 'uniform' , activation = 'sigmoid'))

#Compiling the ANN (adam is an implementation of stochastic gradient descent)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 

# Fitting the ANN to the training set 
classifier.fit(customers, is_fraud, batch_size = 1, nb_epoch = 100)

# Part 3: Making predictions and evaluting the model

# Predicting the probabilities of frauds
# Creates array of customer probabilites of fraud, and then concatenates their ids onto it as well.
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]


