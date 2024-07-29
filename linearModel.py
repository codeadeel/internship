import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss
import json as storage


class ModelUsingLibs:
    """
        A class to manage and analyze housing data using machine learning models in libraries.

        Attributes:
        ----------
        file : DataFrame
            The housing data read from a CSV file.
        featureNames : list
            The names of the features in the dataset.
        scaledFeatures : ndarray
            The features of the dataset after scaling.
        reg : object
            The LinearRegression model object.
        thetas : list
            The coefficients of the linear regression model.
        trainX, trainY, validX, validY, testX, testY : ndarray
            The training, validation, and test datasets and their labels.
    """
    def __init__(self):
        """
            Initializes the ModelUsingLibs class with default values and reads the CSV file.
        """
        self.file = pd.read_csv("E:/progamming/Machine learning/internship/housing2.csv")
        self.featureNames = ["price", "area", "bedrooms", "bathrooms", "stories", "parking", "basement"]
        self.scaledFeatures = []
        # self.X_train = self.X_test = self.y_train = self.y_test = 0
        self.reg = 0
        self.thetas = []
        
        self.trainX = self.trainY = []
        self.validX = []
        self.validY = []
        self.testX = []
        self.testY = []
    
    def readCSV(self):
        """
            Converts the 'basement' column values to numerical format and prints the CSV file.
        """
        self.file['basement'] = self.file['basement'].map({'yes': 1, 'no': 0})
        print(self.file)
        

    def scaleData(self):
        """
            Scales the feature columns using StandardScaler and combines with the 'basement' column.
        """
        scale = StandardScaler()
        X = self.file[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
        self.scaledFeatures = scale.fit_transform(X)
        
        basement_column = self.file['basement'].values.reshape(-1, 1)
        self.scaledFeatures = np.hstack((self.scaledFeatures, basement_column))
        print("Scaled features: ", self.scaledFeatures)
    
    def splittingData(self):
        """
            Splits the data into training, validation, and test sets.
        """
        Y = self.scaledFeatures[:, 0]
        X = self.scaledFeatures[:, 1:6]
        X_train_valid, self.testX, y_train_valid, self.testY = train_test_split(X, Y, test_size=0.15, random_state=42)
        self.trainX, self.validX, self.trainY, self.validY = train_test_split(X_train_valid, y_train_valid, test_size=0.176, random_state=42)
        
        
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    
    def graphBwFeatures(self):
        """
            Plots the relationships between features and the target variable using seaborn.
        """
        df_scaled = pd.DataFrame(self.scaledFeatures, columns = self.featureNames)
        sns.lmplot(x ="area", y ="price", data = df_scaled, ci = None) 
        plt.show()
        sns.lmplot(x ="bedrooms", y ="price", data = df_scaled, ci = None) 
        plt.show()
        sns.lmplot(x ="bathrooms", y ="price", data = df_scaled, ci = None) 
        plt.show()
        sns.lmplot(x ="stories", y ="price", data = df_scaled, ci = None) 
        plt.show()
        sns.lmplot(x ="parking", y ="price", data = df_scaled, ci = None) 
        plt.show()
        sns.lmplot(x ="basement", y ="price", data = df_scaled, ci = None) 
        plt.show()
    
    def calcCost(self):
        """
            Calculates the cost function (Mean Squared Error) for the training data.
            
            Returns:
            -------
            float
                The computed cost value.
        """
        preds = self.reg.predict(self.trainX)
        mse = mean_squared_error(self.trainY, preds)
        mse = mse / 2
        return mse
        
        # preds = self.reg.predict(self.X_train)
        # mse = mean_squared_error(self.y_train, preds)
        # mse = mse / 2
        # return mse

    def training(self):
        """
            Trains the LinearRegression model on the training data and calculates and prints the training cost.
        """
        self.reg = LinearRegression().fit(self.trainX, self.trainY)
        cost = self.calcCost()
        print(f"Cost from training: {cost}")
        
        # self.reg = LinearRegression().fit(self.X_train, self.y_train)
        # cost = self.calcCost()
        # print(f"Cost from training: {cost}")
    
    def validation(self):
        """
            Validates the model on the validation dataset and prints validation loss and accuracy.
        """
        validLoss = self.Loss(self.validX, self.validY)
        validAccu = self.calcAccuracy(self.validX, self.validY)
        print(f"Validation Loss: {validLoss}, Validation Accuracy: {validAccu * 100}")
    
    def testing(self):
        """
            Tests the model on the test dataset and prints testing loss and accuracy.
        """
        testLoss = self.Loss(self.testX, self.testY)
        testAccu = self.calcAccuracy(self.testX, self.testY)
        print(f"Testing Loss: {testLoss}, Testing Accuracy: {testAccu * 100}")
    
    def Loss(self, X, Y):
        """
            Calculates the loss (Mean Squared Error) for a given dataset.
            
            Parameters:
            X : ndarray
                Features of the dataset.
            Y : ndarray
                Target feature(price) of the dataset.
            
            Returns:
            -------
            float
                The computed loss value.
        """
        preds = self.reg.predict(X)
        mse = mean_squared_error(Y, preds)
        mse = mse / 2
        return mse
    
    def calcAccuracy(self, X, Y):
        """
            Calculates the accuracy of the model on a given dataset.
            
            Parameters:
            X : ndarray
                Features of the dataset.
            Y : ndarray
                Target feature(price) of the dataset.
            
            Returns:
            -------
            float
                The computed accuracy value.
        """
        accuracy = self.reg.score(X, Y)
        return accuracy
        
    def thetaValues(self):
        """
            Extracts and prints the model's theta values (intercept and coefficients).
        """
        inter = float(self.reg.intercept_)
        theta = self.reg.coef_
        self.thetas.append(inter) 
        for i in range(len(theta)):
            self.thetas.append(float(theta[i]))
        print(self.thetas) 
    
    def storeJson(self):
        """
            Stores the model parameters in a JSON file.
        """
        data = {'Thetas' : self.thetas, 
                'TrainingLoss' : self.calcCost(),
                'ValidationLoss': self.Loss(self.validX, self.validY), 
                'ValidationAccuracy' : self.calcAccuracy(self.validX, self.validY) * 100,
                'TestingLoss' : self.Loss(self.testX, self.testY),
                'TestingAccuracy' : self.calcAccuracy(self.testX, self.testY) * 100}
        
        with open("ModelWithLibsValues.json", "w") as file3:
            storage.dump(data, file3)

        print("Everything stored successfully!")
    
    def comparison(self):
        """
            Compares the model performance with another model based on stored parameters.
        """
        with open("thetaValueAfterTraining.json", "r") as file4:
            data = storage.load(file4)
        
        othertrainLoss = data['TrainingLoss']
        othervalLoss = data['ValidationLoss']
        othervalidAccu = data['ValidationAccuracy']
        othertestLoss = data['TestingLoss']
        othertestAccu = data['TestingAccuracy']
        
        print(f"\nValidation Accuracy WITH using libraries: {self.calcAccuracy(self.validX, self.validY) * 100}")
        print(f"Validation Accuracy WITHOUT using libraries: {othervalidAccu}\n")
        
        print(f"Testing Accuracy WITH using libraries: {self.calcAccuracy(self.testX, self.testY) * 100}")
        print(f"Testing Accuracy WITHOUT using libraries: {othertestAccu}\n")
        
        print(f"Training Loss WITH using libraries: {self.calcCost()}")
        print(f"Training Loss WITHOUT using libraries: {othertrainLoss}\n")
        
        print(f"Validation Loss WITH using libraries: {self.Loss(self.validX, self.validY)}")
        print(f"Validation Loss WITHOUT using libraries: {othervalLoss}\n")
        
        print(f"Testing Loss WITH using libraries: {self.Loss(self.testX, self.testY)}")
        print(f"Testing Loss WITHOUT using libraries: {othertestLoss}\n\n")
    
        
    
# Creating an instance of ModelUsingLibs and executing the methods    
linear = ModelUsingLibs()
linear.readCSV()
linear.scaleData()
linear.splittingData()
# linear.graphBwFeatures()
linear.training()
linear.validation()
linear.testing()
linear.thetaValues()
linear.storeJson()
linear.comparison()
linear.trainingLogistic()
linear.validationLogistic()
linear.testingLogistic()