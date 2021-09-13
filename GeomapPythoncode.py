
from keras.models import load_model
#from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import shutil 
import os 

model=load_model('C:/Users/Balaji Dhandapani/Downloads/Dreap/GeomapModel_vgg16.h5')

path = r'C:\Users\Balaji Dhandapani\Downloads\Dreap\Inputtestfiles\Geomap'

img_path = r'C:\Users\Balaji Dhandapani\Downloads\Dreap\Inputtestfiles\Geomap\Map11.jpg'

label = ['Geomap','otherfile']
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
ind = np.where(features == 1)[1][0]
print('Predicted Array:',features)
print('Predicted Label:',label[ind])

if ind == 1:
    Destination = 'C:/Users/Balaji Dhandapani/Downloads/Dreap/OtherfileFolder'
    print(Destination)
elif ind == 0 :
    Destination = 'C:/Users/Balaji Dhandapani/Downloads/Dreap/GeoMapFolder'
    print(Destination)

dest = shutil.move(img_path, Destination) 

print("Destination path:", dest) 
