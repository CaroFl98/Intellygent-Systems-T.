#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import warnings
from pylab import savefig
warnings.filterwarnings( "ignore" )
  
# Logistic Regression
class LogitRegression() :
    def __init__( self, learning_rate, iterations ) :        
        self.learning_rate = learning_rate        
        self.iterations = iterations
          
    #Training function
    def fit( self, X, Y ) :              
        self.m, self.n = X.shape        
        # Weight initialization        
        self.W = np.zeros( self.n )        
        self.b = 0        
        self.X = X        
        self.Y = Y
          
        #Gradient descent function          
        for i in range( self.iterations ) :            
            self.update_weights()            
        return self
      
    #Gradient descent update values functiom
    def update_weights( self ) :           
        A = 1/(1+np.exp(-(self.X.dot(self.W)+self.b)))
          
        # calculate gradients        
        tmp = (A-self.Y.T)        
        tmp = np.reshape(tmp,self.m)        
        dW = np.dot(self.X.T,tmp)/self.m         
        db = np.sum(tmp)/self.m 
          
        #Update weight values  
        self.W = self.W-self.learning_rate*dW    
        self.b = self.b-self.learning_rate*db
          
        return self

    #Predict function  
    def predict( self, X ) :    
        Z = 1/(1+np.exp(-(X.dot(self.W)+self.b)))        
        Y = np.where(Z>0.5,1,0)        
        return Y
  
def main():
      
    # Import data base
    path = (r"C:\Virtual Environment\Scripts\wdbc2.csv")
    df = pd.read_csv(path)

    #Clean missing values for all the data set 
    df.dropna()
    #Drop not used columns
    df=df.drop(['id'],axis=1)
    #Change target values to boolean
    df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]

    #Check correllations
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(df.corr(),annot=True,linewidth=0.05,ax=ax, fmt= '.2f', cmap=plt.cm.cool)   
    plt.savefig('BC_HM.png', dpi=400)
    plt.show()
    print("I will delete variables for SE and Worst\n this means that it may be a problem for my result because they are redundant.\nThis also means my model would not be able \nto distinguish between independent and dependent variables.\nAll variables are corelated but we have redundant information \nI will delete standar error and worst variables, so we dont have redundant values")
    
    #Count values
    print("Counting positive and negative values:")
    print(df['diagnosis'].value_counts())

    #Set x and y
    X = df.drop(['diagnosis','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'], axis = 1)
    Y = df.diagnosis.values

    # Splitting dataset into train 80% and test set 20%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 42 )
      
    # Model training    
    model = LogitRegression( learning_rate = 0.01, iterations = 1000 ) 
    model.fit( X_train, Y_train )    
      
    # Prediction on test set
    Y_pred = model.predict( X_test ) 
      
    # measure performance    
    correctly_classified = 0    
      
    # counter    
    count = 0    
    for count in range( np.size( Y_pred ) ) :  
        
        if Y_test[count] == Y_pred[count] :            
            correctly_classified = correctly_classified + 1
            count = count + 1

    print("\nConfusion matrix results for hand test model 20%")
    print(confusion_matrix(Y_test,Y_pred))             
    print( "Accuracy on test set by our model:  ", (correctly_classified / count ) * 100 )
    
    #UserTest
    print("\nAnswer the next questions: \n")
    UserInfo = []
    UserInfo.append(float(input("Radius or mean of distances from center to points on the perimeter: ")))
    UserInfo.append(float(input("Texture or standard deviation of gray-scale values: ")))
    perimeter_mean = float(input("Perimeter: "))
    UserInfo.append(perimeter_mean)
    area_mean = float(input("Area: "))
    UserInfo.append(area_mean)
    UserInfo.append(float(input("Smoothness or local variation in radius lengths: ")))
    UserInfo.append((perimeter_mean)**2/(area_mean)-1)
    UserInfo.append(float(input("Severity of concave portions of the contour: ")))
    UserInfo.append(float(input("Number of concave portions of the contour: ")))
    UserInfo.append(float(input("Symmetry mean: ")))
    UserInfo.append(float(input("Fractal dimension mean: ")))

    UserInfo_df = pd.DataFrame([UserInfo])
    prediction = model.predict(UserInfo_df)
    
    print("Results will be printed with 1 as Breast Cancer case and 0 as healthy")
    print("Hand Prediction: ")
    print(prediction)
 
if __name__ == "__main__" :     
    main()
# %%
