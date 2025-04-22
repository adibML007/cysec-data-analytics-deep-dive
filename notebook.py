#!/usr/bin/env python
# coding: utf-8

# ## CySec data analytics Deep dive lab
# 
# In this lab, we'll continue from our analysis on the cysec data analytics lab. As you have seen in the previous lab, we only looked at accuracy of our models. There are many other performance metrics available for binary classification. We shall look at some of these in this lab.

# ## STEP 1: Import necessary libraries.

# In[ ]:





# ## STEP 2: Load the dataset

# In[ ]:





# ## STEP 3: Load the pre-processed features (X), target (y) and models from previous lab

# In[ ]:





# Answer the following questions based on the plots.
# 1. Which model seems to perform better than others?
# 2. Why do you think it performed better?
# 3. Please explain True positive, true negative, false positive and false negative values for binary classification in this context.

# Enter you answer here and make sure to comment them out.
# 
# 1.
# 2.
# 3.

# ## STEP 4: Visualize the model performance
# 
# In this step, you need to visualize the model performance by plotting the confusion matrix. Know more about it here: https://www.geeksforgeeks.org/confusion-matrix-machine-learning/. Please use StratifiedKFold this time instead of KFold and use 5 splits. 
# 
# Hint: You should have figures like the following for each of the models
# 
# https://github.com/adibML007/cysec-data-analytics-deep-dive/blob/main/Figures/confusion_matrix_LR_fold_5.png
# 

# In[ ]:





# ## STEP 5: TPR and FPR analysis with Operating Characteristics Curve (AUC-ROC)
# In this step, we'll plot True positive rate (TPR) and False positive rate (FPR). These two values are extremely important for almost any ML models. In this plot, the area under the curve represents the AUC-ROC. Know more about it here: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=The%20area%20under%20the%20ROC,curve%20(AUC)%20of%201.0.

# In[ ]:





# In[ ]:





# ## STEP 6: Check feature importance for Random Forest Classifier
# 
# In this step, plot the top 10 most important features with their corresponding importance. Sample plot is here: https://github.com/adibML007/cysec-data-analytics-deep-dive/blob/main/Figures/feat_imp_top10.png
# 

# In[ ]:





# ## STEP 7: Plot a Pareto chart
# There are many ways to show feature importance. One of the most popular ones is Pareto chart. Show the pareto chart that contains the factors whose contributions sum up to 80% like this one: https://github.com/adibML007/cysec-data-analytics-deep-dive/blob/main/Figures/pareto_chart.png
# 

# In[ ]:





# ## STEP 8: Refine the model
# Use only the important features to refine the model. Create multiple subplots to show the results for each fold.
# 
# Hint: You should have a figure like this: https://github.com/adibML007/cysec-data-analytics-deep-dive/blob/main/Figures/all_folds_cm.png
# 

# In[ ]:





# ## STEP 9: Clustering analysis (Optional)
# 
# Do a K-means clustering on the given dataset. Although it is not required for labeled dataset, this is just for practice. We can pretend that the label does not exist. 
# 
# Plot a figure like this with 3 clusters: https://github.com/adibML007/cysec-data-analytics-deep-dive/blob/main/Figures/clusters_3.png

# In[ ]:





# ## STEP 10: Convert your notebook file to a script (.py)

# In[ ]:


get_ipython().system('jupyter nbconvert --to script notebook.ipynb')

