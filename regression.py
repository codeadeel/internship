import matplotlib as mp
import pandas as pan
import math 

class data:
    def __init__(self):
        self.file = pan.read_csv("housing2.csv")
        self.allScaledFeatures = []
        # h(x) = theta0 + (theta1)(x1^2) + (theta2)(x2^3) + (theta3)(ln(x3)) + (theta4)(e^x4) + (theta5)(x5^4)
        self.theta = [0.034, -0.089, 0.067, -0.043, 0.021, -0.056] 
        self.alpha = 0.01
        
    def openCSV(self):
        print(self.file)
        
    def scaling(self, feature):
        features = self.file[feature]
        featureScaled = []
        maxVal = features.max()
        minVal = features.min()
        
        for i in range(len(features)):
            temp = features.iloc[i] - minVal
            temp2 = maxVal - minVal
            if temp2 == 0: 
                print(f"Cannot divide by temp2 as it is zero, so {feature} wasnt scaled") 
                return 
            else:
                featureScaled.append(temp/temp2)
         
        self.allScaledFeatures.append(featureScaled)   
        print("Scaled ", feature, ": ",featureScaled, "\n")
        #print(f"Min: {minVal}, Max: {maxVal}")
        
    def calcHypo(self, exampleNum): # exampleNum -> 0-14
        hypo =  (self.theta[0])*(1) + \
                (self.theta[1])*(math.pow(self.allScaledFeatures[1][exampleNum], 2)) + \
                (self.theta[2])*(math.pow(self.allScaledFeatures[2][exampleNum], 3)) + \
                (self.theta[3])*(math.log(self.allScaledFeatures[3][exampleNum] + 1e-5)) + \
                (self.theta[4])*(math.pow(2.718, self.allScaledFeatures[4][exampleNum])) + \
                (self.theta[5])*(math.pow(self.allScaledFeatures[5][exampleNum], 4))
        #print(hypo)
        return hypo

    def calcCost(self, thetaNum):
        temp3 = 0
        if thetaNum == 0:
             for i in range(15):
                temp3 = self.calcHypo(i) - self.allScaledFeatures[0][i]
        else:
            for i in range(15):
                temp3 = self.calcHypo(i) - self.allScaledFeatures[0][i]     # h(x) - y
                temp3 = temp3 * self.allScaledFeatures[thetaNum][i]
            
        temp3 = temp3 / 15
        
        return temp3
    
    def gradient(self):
        for _ in range(9000):
            tempThetas = []
            for i in range(len(self.theta)):
                temp4 = self.theta[i] - (self.alpha)*(self.calcCost(i))
                tempThetas.append(temp4)    
            self.theta = tempThetas
    
    def MAIN(self): 
        self.openCSV()
        self.scaling("price")
        self.scaling("area")
        self.scaling("bedrooms")
        self.scaling("bathrooms")
        self.scaling("stories")
        self.scaling("parking")
        
        print(f"Thetas before gradient descent: {self.theta}")
        
        self.gradient()
        
        print(f"Thetas after gradient descent: {self.theta}")
        
        
     
# ---MAIN---   
obj = data()
obj.MAIN()