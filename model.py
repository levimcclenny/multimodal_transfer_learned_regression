import keras
import keras.backend as K
import tensorflow as tf
from keras import applications
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dropout, InputLayer, concatenate


def MLP_model(in_dim):
    model = Sequential()
    model.add(Dense(100, input_dim=in_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    return model

# CNN_type codes, 1 = VGG16, 2 = ResNet, 3 = Inception
# note for Inception that input image size is (299,299,3)
def CNN_model(type):
    print(type)
    if type == 1:
        net = applications.vgg16.VGG16(weights='imagenet', input_shape = (224,224,3))
        output = net.layers[-2].output
        # This keeps both 4096 layers and fine tunes after that.
        # After much experimentation, this approach yields better regression accuracy with fewer training parameters,
        # as well as more stable training. Uncomment the following line for removal of the 4096 layers:
        # output = net.layers[-4].output
        # but beware stability issues in training with a small dataset

    elif type == 2:
        print("inside resnet loop")
        print(type)
        net = applications.resnet50.ResNet50(weights='imagenet')
        output = net.layers[-2].output
        #this picks up after the AveragePooling2d layer before the final 1000 class FC layer

    elif type == 3:
        net = applications.inception_v3.InceptionV3(weights='imagenet')
        output = net.layers[-2].output
        # this picks up after the AveragePooling2d layer before the final 1000 class FC layer
        # note that input size must be (299,299,3)

    cnn_model = Model(net.input, output)
    cnn_model.trainable = False
    for layer in cnn_model.layers:
        layer.trainable = False
    print(cnn_model.summary())


    model = Sequential()
    model.add(cnn_model)
    model.add(Dropout(0.4))
    model.add(Dense(1000, activation='relu'))
    return model


def build_model(CNN_type, descriptive_dim, target_dim):
    mlp = MLP_model(in_dim = descriptive_dim)
    cnn_model = CNN_model(type = CNN_type)
    input = concatenate([mlp.output, cnn_model.output])

    x = Dense(1000, activation="relu")(input)
    x = Dropout(0.3)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(target_dim, activation="linear")(x)

    model = Model(inputs=[mlp.input, cnn_model.input], outputs=x)
    #model = Model(inputs=CNN_model.input, outputs=x)
    return model
