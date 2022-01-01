# import dependencies

import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.xception import preprocess_input
from PIL import Image
from utils import local_css
from grad_cam import get_img_array,make_gradcam_heatmap,save_and_display_gradcam
import os


# load model
@st.cache(allow_output_mutation=True)
def loaded_model():
    model = load_model('pneumonia_xception.h5')
    return model

classsifier_model=loaded_model()
last_conv_layer_name = "block14_sepconv2_act"
local_css("images/style.css")

# function for classification of image
st.title("PNEUMONIA CLASSIFICATION")
def predict_image(image_data):
    image=load_img(image_data,target_size=(224,224,3))
    img = img_to_array(image)
    img = img/255.0
    img = np.expand_dims(img,axis=0)
    prediction = classsifier_model.predict(img)
    return prediction

# main function to get tht image and to show the result
def main():
    uploaded_file = st.file_uploader("UPLOAD  A  X-RAY  IMAGE", type=['jpg','jpeg','png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image',width=995)
        save_image_path='images/uploads/'+uploaded_file.name
        with open(save_image_path,'wb') as f:
            f.write(uploaded_file.getbuffer())

        generate_pred = st.button('GENERATE PREDICTION')

        if generate_pred:
            prediction = predict_image(save_image_path)

            print(uploaded_file.name)
            if (prediction[0][0] > 0.5):
                stat = prediction[0][0] * 100
                stat=np.round(stat,2)
                message = "THIS IMAGE IS {}% {}".format(stat,"NORMAL")
                st.title(message)

            else:
                stat = (1.0 - prediction[0][0]) * 100
                stat=np.round(stat,2)
                message = "THIS IMAGE IS {}% {}".format(stat, "PNEUMONIA")
                st.title(message)

            # grad cam functions
                img_array = preprocess_input(get_img_array(save_image_path, size=(224, 224)))
                heatmap = make_gradcam_heatmap(img_array, classsifier_model, last_conv_layer_name)
                save_and_display_gradcam(save_image_path, heatmap)
                st.image('images/uploads/cam.jpg',width=995)


                folder='uploads'
                for filename in os.listdir(folder):
                    file_path=os.path.join(folder,filename)
                    os.unlink(file_path)

if __name__ == "__main__":
    main()



























