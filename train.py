import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import optimizers
from model import build_model
from utils import *

class DMTLR:
   def __init__(self, type):
      self.type = type
      self.target_dim = None
      self.descriptive_dim = None
      self.state = np.random.randint(100)
      self.model = None

   def fit(self, train_imgs, train_vals, train_target_vals, num_epochs):
      self.target_dim = np.shape(train_target_vals)[1]
      self.descriptive_dim = np.shape(par)[1]


      model = build_model(self.type, self.descriptive_dim, self.target_dim)
      print(model.summary())
      model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=.001, decay = .001/50))

      history = model.fit([train_vals, train_imgs], train_target_vals , batch_size = 32,
                    epochs=num_epochs, verbose = 1)

      self.model = model


   def predict(self, test_imgs, test_vals):
      preds = self.model.predict([test_vals, test_imgs])
      return preds



if __name__ == "__main__":

   #grab the CNN type desired from the user
   type, epochs = get_type_input()

   #Get data for reproducing results from paper
   images, par, targets = get_data(type)

   # state is extremely important to set prior to splitting data, as this state tells train_test_split to grab the same images
   # for each test value, corresponding to a specific output. This makes splitting the data drastically easier.
   state = 73

   # use state with constant test_size to grab train/test images with corresponding train/test vectors.
   # since the y (output) vectors will be the same we only need to grab them once
   X_train_imgs, X_test_imgs, _, _ = train_test_split(images, targets, test_size=0.33, random_state=state)
   X_train_vals, X_test_vals, y_train, y_test = train_test_split(par, targets, test_size=0.33, random_state=state)

   dmtlr = DMTLR(type)
   dmtlr.fit(X_train_imgs, X_train_vals, y_train, num_epochs=epochs)
   preds = dmtlr.predict(X_test_imgs, X_test_vals)
   print(np.shape(preds))
   plot_preds(preds, y_test)
