# Credit-Card-Application-Fraud-Detector

Built two models to attempt to perform fraud identitifciation through customer segmentation: 
- built_som, which uses a Self Organizing Map (SOM) structure taken from [Minisom](https://github.com/JustGlowing/minisom) 
to attempt to identify people who were most likely to have commited fraud by visualizing customer segmentation
- case_study_built.py, that combines the above algorithim with an Artificial Neural Network that determines the probability of 
customers having commited fraud and rank orders users by that

# Getting Started
- Download built_som.py, minisom.py, Credit_Card_Applications.csv, and case_study_built.py into the same folder
- Run built_som.py and/or case_study_built.py up to line 60
- Given the SOM that is produced (as it changes everytime), input the customer segments you would like to explore further as
those with fraudlent applications on line 64. It is your discretion what values to explore, in my implementation I chose 
to select those that would be of immediate priority (customer groups that have a likelihood of commiting fraud of above 80% and who had an accepted credit card application).

# Built With
- **The environment:** [Spyder IDE](https://www.spyder-ide.org/)
- **Data Processing:** [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), and [Scikit-learn](https://scikit-learn.org/stable/index.html) 
- **Model Training/Iterating:** [Keras](https://keras.io/) and [Scikit-learn](https://scikit-learn.org/stable/index.html)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/) (specifically [Pylab](https://www.tutorialspoint.com/matplotlib/matplotlib_pylab_module.htm))
- **Education:** Part of the [Deep Learning A-Z Udemy course](https://www.udemy.com/course/deeplearning/)

# Next Steps For Improvement
- Reduce overfitting on my model as it is very much so currently

# Author
- **Rikard Saqe** [Github](https://github.com/rikardsaqe/)

# License
- This project is licensed under the MIT License, see the [LICENSE.txt](https://github.com/rikardsaqe/Credit-Card-Application-Fraud-Detector/blob/master/LICENSE) file for details
