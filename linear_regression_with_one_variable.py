"""
Note:
If you are going to use pandas please type .values after reading the file
ex. df = pd.read_csv('Your_Dataset').values
"""


import numpy as np

class LinearRegression():

#----------------------------------------------------------------------------
#----------------------- Constructor ----------------------------------------
#----------------------------------------------------------------------------

  def __init__(self, alpha = 0.001):
    self.alpha = alpha
    self.training_size = 0
    self.intercept = 0
    self.sum_error = 0
    self.sum_squared_error = 0
    self.sum_error_x = 0
    self.cost_function = 1000
    self.number_features = 0
    self.slope = 0

#----------------------------------------------------------------------------
#----------------------- Count Number Of Features ---------------------------
#----------------------------------------------------------------------------

  def count_features(self, X):
    self.number_features = X.shape[1]

#----------------------------------------------------------------------------
#----------------------- Generate Random Weights ----------------------------
#----------------------------------------------------------------------------

  def generate_random(self):
    number = np.random.randint(1, 10)
    self.slope = number
    self.intercept = number

#----------------------------------------------------------------------------
#----------------------- Predict (Use Model) --------------------------------
#----------------------------------------------------------------------------

  def predict(self, to_be_predicted):
    return self.slope * to_be_predicted + self.intercept

#----------------------------------------------------------------------------
#----------------------- Calculate Error ------------------------------------
#----------------------------------------------------------------------------

  def calculate_error(self, Y_True, Predicted):
    error = Predicted - Y_True
    self.sum_error = sum(error)
    return error

#----------------------------------------------------------------------------
#----------------------- Squared Error --------------------------------------
#----------------------------------------------------------------------------

  def squared_error(self, Y_True, Predicted):
    squared_error = (Predicted - Y_True) ** 2
    self.sum_squared_error = sum(squared_error)
    return squared_error

#----------------------------------------------------------------------------
#----------------------- Error Times X --------------------------------------
#----------------------------------------------------------------------------

  def error_x(self, Y_True, Predicted, X):
    error_x = (Predicted - Y_True) * X
    self.sum_error_x = sum(error_x)
    return error_x

#----------------------------------------------------------------------------
#----------------------- Training Length ------------------------------------
#----------------------------------------------------------------------------

  def training_length(self, X):
    self.training_size = len(X)

#----------------------------------------------------------------------------
#----------------------- Cost Function --------------------------------------
#----------------------------------------------------------------------------

  def cost(self):
    self.cost_function = 1/(2*self.training_size) * self.sum_squared_error

#----------------------------------------------------------------------------
#----------------------- Gradient Descent -----------------------------------
#----------------------------------------------------------------------------

  def fit(self, X_scaled, Y_scaled, print_iteration_count = False):
    self.training_length(X)
    self.generate_random()
    predicted = self.predict(X)
    error = self.calculate_error(Y, predicted)
    squared_error = self.squared_error(Y, predicted)
    error_x = self.error_x(Y, predicted, X)
    self.cost()
    cost_history = [self.cost_function, self.cost_function]
    counter = 0
    while cost_history[-1] <= cost_history[-2]:
      counter += 1
      predicted = self.predict(X_scaled)
      error = self.calculate_error(Y_scaled, predicted)
      squared_error = self.squared_error(Y_scaled, predicted)
      error_x = self.error_x(Y_scaled, predicted, X_scaled)
      self.cost()
      cost_history.append(self.cost_function)
      self.intercept -= (self.alpha/self.training_size) * self.sum_error
      self.slope -= self.alpha/self.training_size * self.sum_error_x
    if print_iteration_count == True:
      print(f'Fitting took {counter} iterations')

X = np.arange(1, 21, 1).reshape(-1, 1)
Y = 2*X + 1

model = LinearRegression(0.01)
model.fit(X, Y)
predicted = model.predict(X)

print(f'Slope: {model.slope}')
print(f'Intercept: {model.intercept}')