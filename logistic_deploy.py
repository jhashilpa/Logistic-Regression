#Let's start with importing necessary libraries
import pickle
import numpy as np
import pandas as pd

class predObj:

    def predict_log(self, dict_pred):
        numeric_cols=['rate_marriage', 'age', 'yrs_married', 'children', 'religious', 'educ']
        # Seggregate the numerical and categorical columns
        print("I/p from the Web API",dict_pred)
        numeric = np.array([])
        categorical = np.array([])
        for index, val in dict_pred.items():
            if index in numeric_cols:
                numeric = np.append(numeric, val)
            else:
                categorical = np.append(categorical, val)

        with open("minMaxScalar.sav", 'rb') as f:
            minMaxScalar = pickle.load(f)

        with open("oneHotEncoder.sav", 'rb') as f:
            oneHotEncoder = pickle.load(f)

        with open("logregModel.sav", 'rb') as f:
            model = pickle.load(f)

        # Transform both numerical and categorical columns
        numArray = minMaxScalar.transform(numeric.reshape(1, -1)).reshape(-1)
        catArray = oneHotEncoder.transform(categorical.reshape(1, -1)).reshape(-1)
        print("MinMax scaling of numerical columns", numArray)
        print("OneHOt encoding of Categorical columns", catArray)

        # Append both the columns so that we can pass the data to the model
        finalInput = np.concatenate((numArray, catArray))
        print("Final Input Array is",finalInput.reshape(1, -1))

        # Predict the above data point
        predictedVal = model.predict(finalInput.reshape(1, -1))
        print("Predicted class is", predictedVal[0], "which means Women will not have affair")


        if predictedVal[0] ==1 :
            result = 'Women will have Affair'
        else:
            result ='Women will not have affair'

        return result



