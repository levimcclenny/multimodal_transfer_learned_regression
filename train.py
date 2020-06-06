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
      self.X_test = None
      self.X_test_imgs = None
      self.y_test = None

   def fit(self):

      #Get data for reproducing results from paper
      images, par, targets = get_data(self.type)

      self.target_dim = np.shape(targets)[1]
      self.descriptive_dim = np.shape(par)[1]

      X_train_imgs, X_test_imgs, y_train, y_test = train_test_split(images, targets, test_size=0.33, random_state=self.state)
      X_train, X_test, y_train, y_test = train_test_split(par, targets, test_size=0.33, random_state=self.state)

      model = build_model(self.type, self.descriptive_dim, self.target_dim)

      model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=.001, decay = .001/50))

      history = model.fit([X_train, X_train_imgs],y_train, batch_size = 32, validation_data = ([X_test, X_test_imgs], y_test),
                    epochs=1, verbose = 1)

      self.X_test = X_test
      self.X_test_imgs = X_test_imgs
      self.y_test = y_test
      self.model = model


   def predict(self):
      preds = self.model.predict([self.X_test, self.X_test_imgs], self.y_test)
      return preds

if __name__ == "__main__":

   #grab the CNN type desired from the user
   type = get_type_input()

   dmtlr = DMTLR(type)
   dmtlr.fit()
   preds = dmtlr.predict()
   plot_preds(preds, dmtlr.y_test)
