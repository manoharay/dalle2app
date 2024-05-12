from PIL import Image
import numpy as np
import cv2
import streamlit as st
from openai import OpenAI
import openai
import os


openai.api_key = os.environ.get('OpenAI_API_Key')


st.set_page_config(layout="wide", page_title="OpenAI DALL-E Example")
st.title("OpenAI DALL-E Example")


def resize_image(image, size):
    # Load the input image
    input_image = cv2.imread(image)

    # Get the dimensions of the input image
    original_height, original_width = input_image.shape[:2]

    # Determine the maximum dimension for resizing
    max_dim = max(original_height, original_width)
    size = size[0]
    # Calculate the scaling factor to resize the image while preserving the aspect ratio
    scale_factor = size / max_dim

    # Resize the image using the scaling factor
    resized_image = cv2.resize(input_image, (0, 0), fx=scale_factor, fy=scale_factor)

    # Get the new dimensions of the resized image
    resized_height, resized_width = resized_image.shape[:2]

    # Calculate the padding needed to make the image 1024x1024
    top_pad = (size - resized_height) // 2
    bottom_pad = size - resized_height - top_pad
    left_pad = (size - resized_width) // 2
    right_pad = size - resized_width - left_pad

    # Pad the image with black spaces
    resized_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.imwrite("resized_image.png", resized_image)

    return resized_image

def create_mask(resized_image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Face detection using a pre-trained model (e.g., Haar cascades or MTCNN)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, minNeighbors=1, minSize=(50, 50))

    # Initialize an empty mask
    mask = np.zeros_like(gray_image)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Calculate the new bounding box dimensions to make the mask smaller
        new_x = int(x-0.1*x)  # Adjust the scaling factor as needed
        new_y = int(y - 0.3 * h)
        new_w = int(1.3* w)
        new_h = int(1.4*h)

        # Generate a mask for the adjusted face ROI
        mask[new_y:new_y+new_h, new_x:new_x+new_w] = 255

    inverted_mask = mask
    cv2.imwrite("mask.png", inverted_mask)
    return inverted_mask

# @st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def create_image(dim, num):
    # OpenAI request
    response = openai.images.edit(
        model="dall-e-2",
        image=open("resized_image.png", "rb"),
        mask=open("mask.png", "rb"),
        prompt="change the background to a forest, and add some magical equipment like a wand and a spellbook for the person, be detailed and realistic",
        n=num,
        size=dim,
    )


    # Get the edited image URL
    image_url = response.data
    return image_url
st.sidebar.title("Settings")

img_file_buffer = st.camera_input("Take a photo")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    image.save("yuha1.png")
    # image = cv2.imread("yuha1.png")
    # img_array = np.array(image)
    # img_array_fin = img_array.astype(np.uint8)
    # cv2.imwrite('yuha2.png', img_array_fin)

with st.sidebar:
    st.write("Upload an image and mask to edit the image.")
    prompt = st.text_area("Prompt", "Change the background to a forest, and add some magical equipment like a wand and a spellbook for the person, be detailed and realistic")
    n = st.number_input("Number of images", 1, 3, 1)
    size = st.selectbox("Image size", ["256x256", "512x512", "1024x1024"], index=2)
    submit = st.button("Submit")
if submit:
    sizes = int(size.split("x")[0])
    resized_image = resize_image("yuha1.png", (sizes, sizes))
    mask = create_mask(resized_image)
    dim = size
    image_url = create_image(dim = dim, num = n) 
    if n > 1:
        for i in range(0, n):
            st.image(image_url[i].url)   
    else:
        st.image(image_url[0].url)
    st.image("resized_image.png")
    st.image("mask.png") 






