import sys, getopt
from pickle import dump, load
import matplotlib.pyplot as plt
import numpy as np

def get_type_input():
   if len(sys.argv[1:]) > 1:
         print('run train.py with only one of either --VGG16 --Inception --ResNet flags')
         sys.exit()
   arg = sys.argv[1:][0].lower()

   if arg == "--vgg16":
      cnn_type = 1
   elif arg == "--resnet":
      cnn_type = 2
   elif arg == '--inception':
      cnn_type = 3
   else:
      print('run train.py with either --VGG16 --Inception --ResNet flags')
   return cnn_type

def get_data(type):
   if type == 1:
      try:
         print("Loading VGG16 images from data/ directory\n")
         images = load(open("data/VGG16_images.pkl", 'rb'))
      except:
         print("VGG16 images couldnt be found in data/VGG16_images.pkl")
   if type == 2:
      try:
         print("Loading ResNet images from data/ directory\n")
         images = load(open("data/ResNet_images.pkl", 'rb'))
      except:
         print("ResNet images couldnt be found in data/ResNet_images.pkl")
   if type == 3:
      try:
         print("Loading Inception images from data/ directory\n")
         images = load(open("data/Inception_images.pkl", 'rb'))
      except:
         print("Inception images couldnt be found in data/Inception_images.pkl")

   try:
      print("loading parameters\n")
      parameters = load(open("data/parameters.pkl", "rb"))
   except:
      print("parameters.pkl couldnt be loaded from data/")

   try:
      print("loading targets\n")
      properties = load(open("data/properties.pkl", "rb"))
   except:
      print("properties.pkl couldnt be loaded from data/")

   return images, parameters, properties


def plot_preds(pred, act, plotline = True):
   fig = plt.figure(figsize=(15,5))
   for i in range(np.shape(pred)[1]):
       plt.subplot(2, 3, i+1)
       if plotline:
          x = np.linspace(-3, 3, 1000)
          plt.plot(x,x, "--", color = 'firebrick', linewidth=3)
       plt.yscale('linear')
       plt.ylabel("$\overline{y}$")
       plt.xlabel("$y_{true}$")
       plt.xlim((-3,3))
       plt.ylim((-5,5))
       plt.scatter(pred[:,i], act[:,i], s=150, color="gray", alpha=0.3)
       #plt.title(names[i])
   fig.tight_layout(pad=2.0)
   plt.show()
