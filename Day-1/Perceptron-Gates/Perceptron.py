import numpy as np

#activation function
def step_function(x):
    return np.where(x>=0,1,0)

#Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        #input sixe os the number of features
        #input size=2 here since features are [x1,x2]
        self.weights=np.zeros(input_size)    #Each weight will be attached to a specific feature. Initially keeping them as 0
        #weights will later be adjusted to improve model's performance by minimising errors
        self.bias=0    #Scalar to shift decision boundary
        self.lr=learning_rate    #Controls how fast weights change when learning
    def predict(self,x):
        linear_output=np.dot(x,self.weights)+self.bias    #x1*w1  +  x2*w2
        return step_function(linear_output)    #the activation function
    #training
    def train(self,x,y,epochs=10):
        for epoch in range(epochs):
            for xi,target in zip(x,y):   #feature vector and target label
                prediction=self.predict(xi)
                update=self.lr*(target-prediction)   #how much did each feature distract from targeting value
                #learning rate is the scaling factor that tells how big a step we can take when we update weights
                #learning rate controls step size
                self.weights+=update*xi    #adjusting weight based on how much its corresponding feature contributed to
                #wrong prediction
                self.bias+=update    #base confidence, doesnt depend on features, shifts decision boundary left or right
