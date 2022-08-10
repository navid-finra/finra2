# finra2

The Firm Centric model is an unsupervised model that ranks firms based on their activities and suggests the items with the highest rank for review by the business. 
We propose two methods for the internal validation of the model. Both methods shall be implemented for validation. The overall strategy for validating the model is as follows:
Method 1: 
We will ask the business to review a minimum of twenty-one (21) firms. The firms that will be reviewed will be selected from the top, middle, and end of the ranking for one representative month, seven (7) of each. Preferably, if we have thirty cases, we can have ten of each reviewed. The review will produce tags that could be used for model validation. The validation process will follow the ML Assurance Strategy, and the information required in the Model Card may be reported in the model card. The items reviewed by the business will be considered as the test set, and validation metrics will be calculated based on that. The methodology of the validation notebook could be used for model validation once we have the test set labels. 

Method 2:
The data could be divided into two homogenous datasets (twin data). To ensure that the twin data are homogenous, the dataset (full data) should be classified into then clusters; the expectation is that the twin data set would have similar distribution across the ten clusters. Then these two datasets will be used for cross-validation. Two models will be developed based on each of the datasets and applied to the other dataset. As such, each of the twin datasets will be once treated as a test set and once as a training set. 
Let's call the two data sets A and B. One model will be developed using A and applied to B, and another will be developed using B and applied to A. Then, the results of the two models on each dataset (A and B) will be compared.  
