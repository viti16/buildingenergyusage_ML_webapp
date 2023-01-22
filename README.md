## Building energy usage prediction using Machine learning using catboost regressor and streamlit to create a web app 

The data is taken from https://www.kaggle.com/competitions/widsdatathon2022/data

### Repo contains following folder/files:
* `Dataset` contains the training and test set
* `Model` contains the trained saved model(.sav) file using catboost inbuilt save model feature
* `Notebook` contains the jupyter notebook that was used to perform machine learning task
* `streamlitapp.py` file contains the commands to create the webpage for the app

## General pipeline for ML task
* Clean and understand the data features
* Performed EDA on the features to find useful features, categorical features and their distribution, correaltions, linear/nonlinear relation
* Used KNN imputer for imputing missing values
* Data is overfitted when modeled using all features so did Feature engineering to remove unnecessary features or combine features to create new useful features
* Used catboost regressor decision tree based model to fit the data
* Used feature importance to identify important features and repeated above steps to reduce overfitting
* Used hyperparameter tuing to get the best parameters for number of decision tress and regularization parameter
* Saved the best model for predicting for the web app

## Create webapp
* Used streamlit app building commands to create a web interface and used saved model for predicting the new data
* Created necessary files for hosting the web app on the streamlit cloud
* Put everything into GitHub repo and uploaded it on the streamlit cloud
 
 ## App is up and running on this link: 
 https://viti16-buildingenergyusage-webapp-streamlitapp-t1snqn.streamlit.app/
