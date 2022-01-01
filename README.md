
# Pneumonia Diagnosis using X-ray

## Introduction

The data is from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

There are two category in kaggle's data sets : Normal and Pneumonia

The data is  split into a set of  3 folders : train val and test


## Images

![App Screenshot](https://github.com/xx-CRAZINESS-xx/Pneumonia-Detection/blob/main/images/folder%20image.png)

### Data Exploration

* The train folder totally have 5216 jpg files (Normal:1341，PNEUMONIA:3875)

* The val folder totally have 16 jpg files (Normal:8，PNEUMONIA:8)

* The test folder totally have 624 jpg files (Normal:234，PNEUMONIA:390)

**Remark\! Data sets for Normal & Pnuemonia are imbalanced (about 1:3)**

### Data Augmentation

```
ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
```

## Screenshots
**If you are not getting the app just like the below picture, then change the streamlit 
theme**

#### Normal
![App Screenshot](https://github.com/xx-CRAZINESS-xx/Pneumonia-Detection/blob/main/images/normal.png)

#### Pneumonia
![App Screenshot](https://github.com/xx-CRAZINESS-xx/Pneumonia-Detection/blob/main/images/pneumonia.png)
## Deployment

Deployed this project on AWS

http://13.233.229.207:8501/


## Model Building


Here I used Xception 

####  Xception Architecture
![App Screenshot](https://github.com/xx-CRAZINESS-xx/Pneumonia-Detection/blob/main/images/xception.png)

**This is a generic code if you want you can use any other model.
All you have to do is replace Xception with your desired model name and last_conv_layer_name with that model's last convolution layer**

```
IMAGE_SIZE=[224,224]
base_model=Xception(input_shape=IMAGE_SIZE + [3],weights='imagenet',include_top=False)

last_conv_layer_name = "block14_sepconv2_act"

for layer in base_model.layers[:-8]:
    layer.trainable=False

```

```
new_model = base_model.output
new_model = GlobalAveragePooling2D()(new_model)
new_model = Dense(2,activation='softmax')(new_model)

model=Model(base_model.input,new_model)

```
### Model training

* batch size = 64
* optimizer = adam
* loss =  categorical_cross_entropy
* epochs = 30 
* steps per epoch = 32

 

## Results
#### Model Accuracy - 93%
![App Screenshot](https://github.com/xx-CRAZINESS-xx/Pneumonia-Detection/blob/main/images/model_accuracy.png)

#### Classification Report
![App Screenshot](https://github.com/xx-CRAZINESS-xx/Pneumonia-Detection/blob/main/images/classification_report.png)

#### ROC 
![App Screenshot](https://github.com/xx-CRAZINESS-xx/Pneumonia-Detection/blob/main/images/roc.png)

## Run Locally

Clone the project

```
  git clone https://github.com/xx-CRAZINESS-xx/Pneumonia-Detection.git
```

Go to the project directory

```
  cd Pneumonia-Detection
```

Install dependencies

```
  pip install -r requirements.txt
```

Start the server

```
  streamlit run app.py
```

