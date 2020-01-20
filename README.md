# Credit-Card-Application-Fraud-Detector

Built two models, built_som, which uses a Self Organizing Map (SOM) structure taken from https://github.com/JustGlowing/minisom  
to attempt to identify people who were most likely to have commited fraud by visualizing customer segmentation, and another,
case_study_built.py, that combines this algorithim with an Artificial Neural Network that determines the probability of 
customers having commited fraud and rank orders users by that.

Both developed in the Spyder IDE using NumPy, Pandas and Scikit-learn for data processing, Keras and Scikit-learn for 
training/iterating on model, and Pylab for visualizing the results.

Important note: SOMs produce a different result each time, so if you wish to expirment with the code you must only run up to 
line 60 in both models and then update line 64 given the SOM produced. It is your discretion what values to put into 
line 64 (which determines which groups of customers are likely to have commited fraud), in my implementation I chose 
to select those that would be of immediate priority (customer groups that have a likelihood of commiting fraud of above 80% and 
who had an accepted credit card application).

I acheived an extremely high accuracy in identifying frauds (~99%), which suggests to me my model is extremely overfitted to 
my data. My next steps on this project would be to attempt to gather more relevant data I can train, and also to train my 
ANN with many iterations of my SOM in hopes of drawing out some consistent trends.

This project was built as part of the Deep Learning A-Z Udemy course: https://www.udemy.com/course/deeplearning/
