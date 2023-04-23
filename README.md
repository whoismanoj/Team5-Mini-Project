# Mini-Project-

project_HBD(All).ipynb is the original ipynb done by SC1015 PT1 Group 5. In it contain all the methods, trials and other ideas that we plan to do. We decided to break into 

The project_HBD_EDA.ipynb is for us to have a better visualisation of the data before we start to decide which variables we want to remove. 

project_HBD(Final).ipynb is the training models we test and use for our problem. We first use linear regression to train with 75% of the dataset and 25% to as test of the dataset. 

The dataset we use are from https://data.gov.sg/dataset/resale-flat-prices, ranging from 1990 to 2023 Apri, totaling up to 895319 data pointers. The dataset excel file we compile and use is too big to fit onto github. Therefore the excel files will be spilt into the years that were provided by the website. The file of 2000 year to 2012 year could not be uploaded to github due to size limit.

We start of with our problem definition, which is to predict the housing resale price based on a few factors from the dataset such as: 1) Accessibility; 2) Remaining lease years; 3) Distance to town and 4) Flat type.

What we access as accessibility is the distance to the nearest MRT station, the remaining lease years will be calculated as by the lease commence date to 99 years, the distance to town, we set to be at the Central Business District (CBD) Latitude and Longitude to be at 1°16'58.8"N 103°51'04.7"E and lastly the flat type will include the storey range, floor area, and flat models.

As we know, the accessibility of a house has a significant impact on its price. Considering that MRT is Singapore’s most common mode of transport and Singapore’s urban planning. We introduced additional factors which are the distance of the house to the nearest MRT station, and distance of the house to the Raffles Place MRT which we define as the CBD.

We achieve this by first compiling a list of MRT stations and retrieving their GeoLocation using API. Then extract the address of each house transaction and convert the addresses to their Geolocation as well. Lastly, we calculated the minimum distance from the house to the nearest MRT station and the distance of the houses to Raffles MRT station.

We also converted the storey_range to storey_median so that we can use this value to train the model. We also used the ‘pd.get_dummies’ function to convert the following categorical variable into dummies variables. Such as 'Town','flat_type','storey_range','flat_model'

We used linear regression, Ordinary least squares (OLS) regression, Sequential model from Tensor flow together with MinMaxScaler

In TensorFlow Sequential models, loss and val_loss refer to the values of the loss function during training and validation, respectively. To feed these data to train our neural network model. We first need to encode the categorical data into a numerical representation. Next, we will use a MinMax scaler to transform the dataset. This scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one. Many machine learning algorithms perform better or converge faster when features are on a relatively similar scale and/or close to normally distributed. During training, the model tries to minimize a loss function that measures the error between the predicted output and the true output for a given input. The choice of loss function depends on the problem being solved, and different loss functions have different properties and behaviors. The loss value represents the average loss over all training samples in a single epoch. The goal of training is to minimize this value over the course of the training process. During validation, the val_loss value is computed by evaluating the model on a separate dataset that the model has not seen before. This dataset is typically used to assess the generalization performance of the model and to avoid overfitting. The val_loss value is used to determine when to stop the training process or adjust the hyperparameters of the model. The loss and val_loss are two key metrics used to evaluate the performance of a TensorFlow Sequential model during training and validation, respectively. The goal of training is to minimize the loss value, while ensuring that the val_loss value remains low to achieve good generalization performance.

Insight of what we have the recommendation will be to incorporate inflation rate to the data so that it will be able to reflect more accurately to the real world. The Neural Network model could be train longer with more nodes and and layer as the amount of data is very large, so as to let it be the same as the real world. We can make use of a lot different types of models or Neural Network to improve the AI to be as real as possible. Other factors like Covid-19, population growth and government policies. 
