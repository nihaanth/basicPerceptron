class Perceptron:
  def __init__(self,eta,epochs):
    np.random.seed(42)
    self.weights = np.random.randn(3) * 1e-4 ##to get small values
    print(f'initial weights without training :{self.weights}')
    self.eta = eta  #learning rate
    self.epochs = epochs


  def activationFunction(self,inputs,weights):
    z = np.dot(inputs,weights)# z =  W * X
    return np.where(z > 0,1,0)#CONDITION , if true ,else false



  def fit(self,X,y):  ##X -- > input var,y -- > lables
    self.X = X
    self.y = y

    X_with_bias = np.c_[self.X,-np.ones((len(self.X),1))]##length of X -- rows - gives array

    print(f'X with bias:{X_with_bias}')

    for epoch in range(self.epochs):
      print("--"*10)#seperator
      print(f'for epoch: {epoch}')
      print("--"*10)

      y_hat = self.activationFunction(X_with_bias,self.weights)##forward propagation

      print(f'predicted value after forward pass:{y_hat}')

      self.error = self.y - y_hat
      print(f'error:\n{self.error}')

      self.weights = self.weights + self.eta * np.dot(X_with_bias.T,self.error)#X_with_bias.T -- > transpose of X_with_bias -- backWARD PROPAGATION
      print(f'updated weights after epoch:{epoch}/{self.epochs}:{self.weights}')

      print('#######'*10)


  def predict(self,X):
    X_with_bias = np.c_[X,-np.ones((len(self.X),1))]
    return self.activationFunction(X_with_bias,self.weights)

  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f'total loss: {total_loss}')
    return total_loss
