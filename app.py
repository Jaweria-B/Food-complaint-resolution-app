import streamlit as st
import os
from clarifai.client.model import Model
import cv2
from urllib.request import urlopen
import numpy as np
from clarifai.modules.css import ClarifaiStreamlitCSS
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont
import base64
from dotenv import load_dotenv

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

load_dotenv()
import os

# Passing the key values
clarifai_pat = os.getenv("CLARIFAI_PAT")


# 1. Function to choose food item for complaint 
def chooseFoodItem():
    options = ['Pita Gyro', 'Coke 250 ml', 'Choco chip cookie', 'ice cream']
    selected_option = st.radio("Select an option:", options)
    return selected_option


# 2. Input description and food images from user
def takeImage():
    uploaded_file = st.file_uploader("Choose a image", type = ['jpg', 'png'])
    
    if uploaded_file is not None:   
        food_item_img = uploaded_file.getvalue()
        st.image(food_item_img)
        st.success("Image Uploaded Successfully")
        return food_item_img
    else:
        st.error("Please upload the image in jpg or png formate")

# Check category and input
def category_match():
    pass

# 5. Recognize the food items in the picture
def foodItemRecognition(food_img):

    PAT = clarifai_pat
    USER_ID = 'clarifai'
    APP_ID = 'main'

    MODEL_ID = 'food-item-recognition'
    MODEL_VERSION_ID = '1d5fd481e0cf4826aa72ec3ff049e044'

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + PAT),)

    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=food_img
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(post_model_outputs_response.status)
        # raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

    # Since we have one input, one output will exist here
    output = post_model_outputs_response.outputs[0]

    food_items = []

    for concept in output.data.concepts:
        # print("%s %.2f" % (concept.name, concept.value))
        food_items.append(concept.name)
    print(food_items)

    return food_items


# 6. Using GPT4-Turbo to test the input and image  
def item_test(item_category, input_item_names):

    prompt = f"You are an expert in making food. I have passed you an array of food items {input_item_names} and a category{item_category}, tell me if any of the food item in the list is used in the creation of the category item. If any food item can be used to make the category item, return a yes. Please give ouptput in boolean form. only use yes or no."

    # Setting the inference parameters
    inference_params = dict(temperature=0.2, max_tokens=103)

    # Using the model GPT-4-Turbo for predictions
    # Passing the image-url and inference parameters
    model_prediction = Model("https://clarifai.com/openai/chat-completion/models/gpt-4-turbo").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)

    # Output
    print(model_prediction.outputs[0].data.text.raw)
    return model_prediction.outputs[0].data.text.raw

# 
def checkBack(image, description):

    prompt = f"Does the image match the description as provided by the user, \
    Description: {description}, \
    list = ['You have been offered 100% cashback','You have been offered 80% cashback','Please provide another image of damaged food parcel or food']\
    1 = If the image match the discription answer for this as yes and no, \
    If the description matches and it's highly damaged then put 2 = 1st element in 'list',\
    If the description matches and it's lightly damaged then put 2 = 2st element in 'list',\
    If the food parcel is not damaged in the image then put 2 = 3rd element in the 'list',\
    Generate only a list with two elements: [1, 2] where 1 and 2 comes from above"

    # prompt = "describe the image"

    # with open(image_path, "rb") as f:
    base64image = base64.b64encode(image).decode('utf-8')

    inference_params = dict(temperature=0.2, max_tokens=100, image_base64=base64image)

    model_prediction = Model("https://clarifai.com/openai/chat-completion/models/gpt-4-vision").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)

    print(model_prediction.outputs[0].data.text.raw)

def main():
    st.set_page_config(page_title="Food Complaint System", layout="wide")
    st.title("Food Complaint Resolution")

    with st.sidebar:
        selected_category = chooseFoodItem() 
        # Text statement
        st.subheader(f"Enter your complaint and upload the images of damaged")
        description = st.text_area("Enter your complaint:", height=100)
        food_item_img = takeImage()

    st.success(f"Category selected: {selected_category}")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Recognition")

        # if food_item_img or description:
        with st.spinner("Generating image..."):
            food_items = foodItemRecognition(food_item_img)
            # if food_items:
            #     st.success("Image generated!")
            # else:
            #     st.error("Failed to generate image.")
        match = item_test(selected_category,  food_items)

        if match:
            checkBack(food_item_img, description)
        else:
            st.write("We'll make sure to send you an agent.")
            

    with col2:
        st.header("Col # 2")
       
    

if __name__ == "__main__":
    main()