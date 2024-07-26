import matplotlib.pyplot as mp
import pandas as pan
import json as storage
import math 

class data:
    def __init__(self):
        self.file = pan.read_csv("housing2.csv")
        self.allScaledFeatures = []
        self.featureNames = ["price", "area", "bedrooms", "bathrooms", "stories", "parking", "basement"]
        # h(x) = theta0 + (theta1)(x1) + (theta2)(x2) + (theta3)(x3) + (theta4)(x4) + (theta5)(x5) + (theta6)(x6)
        self.theta = [0.034, -0.089, 0.067, -0.043, 0.021, -0.056, 0.035] 
        self.alpha = 0.0001
        self.epochs = 50
        self.totalLen = 0
        self.trainingLen = 0
        self.trainingData = []
        self.validLen = 0
        self.validData = []
        self.testLen = 0
        self.testData = []
        self.batches = []
        
    def openCSV(self):
        print(self.file)
        self.totalLen = len(self.file)
        self.trainingLen = int(0.7 * self.totalLen)
        self.validLen = self.testLen = int(0.15 * self.totalLen)
        
        #print(f"Total: {self.totalLen}, Training: {self.trainingLen}, Valid: {self.validLen}, Test: {self.testLen}")
        
    def seperateData(self):
        self.trainingData = [
            {self.featureNames[ind]: self.allScaledFeatures[ind][i] for ind in range(len(self.featureNames))}
            for i in range(self.trainingLen)
        ]
        
        self.validData = [
            {self.featureNames[ind]: self.allScaledFeatures[ind][i] for ind in range(len(self.featureNames))}
            for i in range(self.trainingLen, (self.trainingLen + self.validLen))
        ]
        
        self.testData = [
            {self.featureNames[ind]: self.allScaledFeatures[ind][i] for ind in range(len(self.featureNames))}
            for i in range((self.trainingLen + self.validLen), self.totalLen)
        ]
        
    def scaling(self, feature):
        features = self.file[feature]
        featureScaled = []
        number = len(features)
        mean = 0
        dev = 0
        if features[0] == "yes" or features[0] == "no":
            for i in range(number):
                if features[i] == "yes":
                    featureScaled.append(1)
                elif features[i] == "no":
                    featureScaled.append(0)
        else:
            for i in range(number):
                mean += features.iloc[i]
            mean = mean / number
            
            for j in range(number):
                dev += math.pow((features.iloc[j] - mean), 2)
            dev = dev / (number - 1)
            dev = math.sqrt(dev)
            
            for k in range(number):
                temp = (features.iloc[k] - mean) / dev
                featureScaled.append(temp)
            
        self.allScaledFeatures.append(featureScaled)  
        
        # maxVal = features.max()
        # minVal = features.min()
        # for i in range(len(features)):
        #     temp = features.iloc[i] - minVal
        #     temp2 = maxVal - minVal
        #     if temp2 == 0: 
        #         print(f"Cannot divide by temp2 as it is zero, so {feature} wasnt scaled") 
        #         return 
        #     else:
        #         featureScaled.append(temp/temp2)
        # print("Scaled ", feature, ": ",featureScaled, "\n")
        #print(f"Min: {minVal}, Max: {maxVal}")
        
    def featureRelations(self, featureX, featureY):
        X = [exampleNum[self.featureNames[featureX]] for exampleNum in self.trainingData]
        Y = [exampleNum[self.featureNames[featureY]] for exampleNum in self.trainingData]
        
        mp.figure(figsize=(10, 5))
        mp.scatter(X, Y)
        mp.xlabel(f"{self.featureNames[featureX]}")
        mp.ylabel(f"{self.featureNames[featureY]}")
        mp.title(f"Relationship between {self.featureNames[featureX]} and {self.featureNames[featureY]}")
        mp.grid(True)
        mp.show()
        
    def createBatches(self):
        batchSize = int(0.1 * self.trainingLen)
        self.batches = [self.trainingData[i : i + batchSize] for i in range(0, self.trainingLen, batchSize)]
        if len(self.batches[-1]) < batchSize:
            lastBatch = self.batches.pop()
            self.batches[-1].extend(lastBatch)
        
    def calcHypo(self, exampleNum):
        hypo =  (self.theta[0])*(1) + \
                (self.theta[1])*(exampleNum[self.featureNames[1]]) + \
                (self.theta[2])*(exampleNum[self.featureNames[2]]) + \
                (self.theta[3])*(exampleNum[self.featureNames[3]]) + \
                (self.theta[4])*(exampleNum[self.featureNames[4]]) + \
                (self.theta[5])*(exampleNum[self.featureNames[5]]) + \
                (self.theta[6])*(exampleNum[self.featureNames[6]])
        #print(hypo)
        return hypo

    def calcCost(self, thetaNum):
        temp3 = 0
        rows = len(self.trainingData)
        if thetaNum == 0:
             for exampleNum in self.trainingData:
                error = self.calcHypo(exampleNum) - exampleNum[self.featureNames[0]]
                temp3 += error
        else:
           for exampleNum in self.trainingData:
                error = self.calcHypo(exampleNum) - exampleNum[self.featureNames[0]]  # h(x) - y
                temp3 += error * exampleNum[self.featureNames[thetaNum]]
            
        temp3 = temp3 / rows
        
        return temp3
    
    def calcValidLoss(self):
        temp = 0
        for exampleNum in self.validData:
            temp += ((self.calcHypo(exampleNum)) - exampleNum[self.featureNames[0]]) ** 2
        temp = temp / (2 * len(self.validData))
        
        return temp
    
    def calcAccuracy(self):
        correct_predictions = 0
        total_predictions = len(self.validData)
        for exampleNum in self.validData:
            prediction = self.calcHypo(exampleNum)
            actual = exampleNum[self.featureNames[0]]
            if abs(prediction - actual) < 0.5:
                correct_predictions += 1
        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy
    
    def gradient(self):
        epochLosses = []
        Accuracies = []
        validLoss = []
        for epoch in range(self.epochs):
            loss = 0
            batchLosses= []
            for batch in self.batches:
                # batch_data = batch.reset_index(drop=True)
                # self.trainingData = batch_data
                temp4 = 0
                temp5 = 0
                for _ in range(100):
                    tempThetas = []
                    for i in range(len(self.theta)):
                        temp4 = self.theta[i] - (self.alpha)*(self.calcCost(i))
                        tempThetas.append(temp4)    
                    self.theta = tempThetas
                    
                for exampleNum in batch:
                    temp5 += (self.calcHypo(exampleNum) - exampleNum[self.featureNames[0]]) ** 2
                temp5 = temp5 / (2 * len(batch))
                batchLosses.append(temp5)
            loss = sum(batchLosses) / len(batchLosses)
            epochLosses.append(loss)
            
            val_loss = self.calcValidLoss()
            validLoss.append(val_loss)
            val_accuracy = self.calcAccuracy()
            Accuracies.append(val_accuracy)
            print(f"Epoch {epoch + 1}, Training Loss: {loss}, Validation Accuracy: {val_accuracy}, validation loss: {val_loss}")
        self.plotLoss(epochLosses, validLoss)
        self.plotAccuracy(Accuracies)
        
    def plotLoss(self, losses, loss2):
        mp.figure(figsize=(10, 5))
        mp.plot(losses, label='Training Loss')
        mp.plot(loss2, label='Validation Loss')
        mp.xlabel('Epochs')
        mp.ylabel('Loss')
        mp.title('Loss over Epochs')
        mp.legend()
        mp.show()
        
    def plotAccuracy(self, accuracy):
        mp.figure(figsize=(10, 5))
        mp.plot(accuracy, label='Validation Accuracy')
        mp.xlabel('Epochs')
        mp.ylabel('Accuracy')
        mp.title('Accuracy over Epochs')
        mp.legend()
        mp.show()
        
    def storeParas(self):
        data = self.theta

        with open("thetaValueAfterTraining.json", "w") as file2:
            storage.dump(data, file2)

        print("Thetas stored successfully after training!")
    
    def loadParas(self):
        with open("thetaValueAfterTraining.json", "r") as file3:
            self.theta = storage.load(file3)
            
        print("theta from json file: ", self.theta, "\n")
        
    def testModel(self):
        correct_predictions = 0
        total_predictions = len(self.testData)
        mse = 0
        for exampleNum in self.testData:
            prediction = self.calcHypo(exampleNum)
            actual = exampleNum[self.featureNames[0]]
            if abs(prediction - actual) < 0.5:
                correct_predictions += 1
        accuracy = (correct_predictions / total_predictions) * 100
        
        for exampleNum in self.testData:
            mse += (self.calcHypo(exampleNum) - exampleNum[self.featureNames[0]]) ** 2
        mse = mse / (2 * len(self.testData))
        rmse = math.sqrt(mse)
        
        print(f"Accuracy after testing data: {accuracy}")
        print(f"MSE after testing data: {mse}")
        print(f"RMSE after testing data: {rmse}")
        
    def userIn(self):
        area = int(input("Enter Area: "))
        beds = int(input("Enter Area: "))
        baths = int(input("Enter Area: "))
        stories = int(input("Enter Area: "))
        parking = int(input("Enter Area: "))
        basement = int(input("Enter Area: "))
        
        hypo =  (self.theta[0])*(1) + \
                (self.theta[1])*(area) + \
                (self.theta[2])*(beds) + \
                (self.theta[3])*(baths) + \
                (self.theta[4])*(stories) + \
                (self.theta[5])*(parking) + \
                (self.theta[6])*(basement)
        
    def MAIN(self): 
        self.openCSV()
        self.scaling("price")
        self.scaling("area")
        self.scaling("bedrooms")
        self.scaling("bathrooms")
        self.scaling("stories")
        self.scaling("parking")
        self.scaling("basement")
        
        self.seperateData()
        
        # self.featureRelations(1, 0)     #since its a 2D list, so we are referencing the features by their list number in allScaledFeature list
        # self.featureRelations(2, 0)
        # self.featureRelations(3, 0)
        # self.featureRelations(4, 0)
        # self.featureRelations(5, 0)
        # self.featureRelations(6, 0)
        
        print(f"Thetas before gradient descent: {self.theta}")
        
        self.createBatches()        
        #self.gradient()
        
        #self.storeParas()
        
        self.loadParas()
        self.testModel()
        
        #print(f"Thetas after gradient descent: {self.theta}")
        
        
     
# ---MAIN---   
obj = data()
obj.MAIN()