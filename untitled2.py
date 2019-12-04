from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from glob import glob 

train_path = "C:/Users/leyla/Desktop/Fruits/fruits-360/Training/"
test_path = "C:/Users/leyla/Desktop/Fruits/fruits-360/Test/"
