## ML Documentation :bookmark_tabs:

**Resources :bookmark:**
Dataset from [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

Link colab [here](https://colab.research.google.com/drive/1wzodQrYE4z4Sn-6pN3c4XYT317w_9X_w?usp=sharing)

**Data preparation**
1. import libraries
```
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
```
2. mount drive
3. identify and remove duplicates
4. data cleaning
5. exploratory data analysis

### Record label names
```
labels = ['Actinic Keratoses', 
          'Basal Cell Carcinoma', 
          'Benign Keratosis', 
          'Dermatofibroma', 
          'Melanocytic Nevi', 
          'Melanoma', 
          'Vascular Skin Lesions']
```
6. split train, validation and test test
- train: 8012 images
- validation: 1001 images
- test: 1002 images
7. data augmentation and flow data
8. visualize data `(20, 224, 224, 3)`

**create the model** 
1. using MobileNetV2 and add top layers network
2. compiling model using adam optimizer and categorical crossentropy loss
3. create class weight to solve imbalance data
```
class_weight = {0 : 1.524759092316784, 
                1 : 1.0724959977588722, 
                2 : 1.0, 
                3 : 2.5697871348507872, 
                4 : 1.0, 
                5 : 1.0, 
                6 : 2.3588922056127766,}
```
> labels class
```
#0 : akiec     327 images
#1 : bcc       514 images
#2 : bkl      1099 images
#3 : df        115 images
#4 : mel      1113 images
#5 : nv       6705 images
#6 : vasc      142 images
```
4. train the model with 50 epoch
5. visualize training result
6. save model
```
model.save('model.h5')
```
```
#Save model with not saving optimizer state when it was last saved so this is save storage and simplify the deployment process
model.save('model_without_optimizer.h5', include_optimizer=False)
```
**model evaluation**
```
The Accuracy of Model is {:.f} 0.7604790329933167
The Loss of Model is {:.f} 0.7357981204986572
```
8. confusion matrix
9. make plot from confusion matrix
10. classification report visualizer displays the Precision, Recall, F1, and Support Scores for the Model


