import streamlit as st
import time 
import cv2
import os
from PIL import Image
from collections import Counter
import pandas as pd
from PIL import Image as PILImage
from torchvision.transforms.functional import to_pil_image
from numpy import asarray
from sahi.utils.yolov8 import download_yolov8n_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict, visualize_object_predictions
from IPython.display import Image as IPythonImage

st.set_page_config(layout="wide", page_title="Arthropods Detection APP")

st.write("## Detect Arthropods from your image")
st.write("Try uploading an image to detect the arthropods. Full quality images can be downloaded from the sidebar:grin:")
st.sidebar.write("## Upload :gear:")


def process_image(image_file, confidence_threshold):
       
    # Define the directory to save the uploaded image
    save_dir = '/content/drive/MyDrive/Capstone_new/Data_New_images/'
    
    # Save the uploaded image to the specified directory
    saved_image_path = os.path.join(save_dir, image_file.name)
    with open(saved_image_path, 'wb') as f:
        f.write(image_file.getbuffer())
    
    # Now, you can use saved_image_path for further processing
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='/content/drive/MyDrive/Capstone_new/runs/detect/train22/weights/best.pt',
        confidence_threshold=confidence_threshold,
        device='cpu'
    )

    result = get_sliced_prediction(
        saved_image_path,
        detection_model=detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3
    )

    img = cv2.imread(saved_image_path, cv2.IMREAD_UNCHANGED)
    img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    numpydata = asarray(img_converted)

    visualize_object_predictions(
        numpydata,
        object_prediction_list=result.object_prediction_list,
        output_dir='/content/drive/MyDrive/Capstone_new',
        file_name='result_701',
        export_format='jpeg'
    )
    col1.write("Original Image :camera:")
    col1.image(saved_image_path)
    result_image_path = '/content/drive/MyDrive/Capstone_new/result_701.jpeg'
    result_image = PILImage.open(result_image_path)
    # col2.write("Detected Image :beetle:")
    # col2.image(result_image)
    #st.image(result_image, caption='Result')
    object_prediction_list = result.object_prediction_list
    class_map = {
    'MA': 'Melon Aphid',
    'NT': 'Nesidiocorius Tenius',
    'OI': 'Orius Insidiosus',
    'WFT': 'Western Flower Thrips',
    'WF': 'White Fly',
    'TS': 'Two Spotted Spidermite'}
    class_counts = {class_name: 0 for class_name in class_map.keys()}
    for prediction in object_prediction_list:
      class_name = prediction.category.name
      if class_name in class_counts:
        class_counts[class_name] += 1
      
    data = {
    'Abbreviation': [],
    'Full Name': [],
    'Count': []}

    for class_name, count in class_counts.items():
      data['Abbreviation'].append(class_name)
      data['Full Name'].append(class_map.get(class_name, class_name))
      data['Count'].append(count)

    df = pd.DataFrame(data)
    
    # Display class names and counts as table
    #table_data = {"Class Name": list(class_counts.keys()), "Count": list(class_counts.values())}
    table_data = {"Abbreviation": df['Abbreviation'].tolist(), "Full Name": df['Full Name'].tolist(), "Count": df['Count'].tolist()}
    col2.write("## Class Counts")
    # col2.table(table_data)
    col2.write(df.set_index('Abbreviation', drop=True))
    col3.write("Detected Image :beetle:")
    col3.image(result_image)

MAX_FILE_SIZE = 5 * 1024 * 1024 
col1, col2, col3 = st.columns(3)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.35, 0.95, 0.35, 0.05,key="confidence_slider")

if st.button('Detect Arthropods'):
    
    
    if my_upload is not None:
        if my_upload.size > MAX_FILE_SIZE:
             st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
        else:
            with st.spinner(text='In progress'):
              process_image(image_file=my_upload,confidence_threshold=confidence_threshold)
            with st.spinner(text='In progress'):
                time.sleep(3)
                st.success('Done')

                bar = st.progress(50)
                time.sleep(3)
                bar.progress(100)
                st.success('Success message')
    else:
        st.error('Please upload an image')
ColorMinMax = st.markdown(
    '''<style> 
    div.stSlider > div[data-baseweb="slider"] > div[data-testid="stTickBar"] > div {
        background: rgb(1 1 1 / 0%);
    } 
    </style>''', unsafe_allow_html=True)

Slider_Cursor = st.markdown(
    '''<style> 
    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
        background-color: #1E90FF; 
        box-shadow: rgba(14, 38, 74, 0.2) 0px 0px 0px 0.2rem;
    } 
    </style>''', unsafe_allow_html=True)

Slider_Number = st.markdown(
    '''<style> 
    div.stSlider > div[data-baseweb="slider"] > div > div > div > div {
        color: #000000; 
    } 
    </style>''', unsafe_allow_html=True)

col = f'''<style> 
    div.stSlider > div[data-baseweb="slider"] > div > div {{
        background: linear-gradient(to right, #1E90FF 0%, #1E90FF {confidence_threshold }%, rgba(151, 166, 195, 0.25) {confidence_threshold }%, rgba(151, 166, 195, 0.25) 100%);
    }} 
    </style>'''

ColorSlider = st.markdown(col, unsafe_allow_html=True)

if confidence_threshold > 0.35 and confidence_threshold < 1.00:

    if my_upload is not None:
          if my_upload.size > MAX_FILE_SIZE:
              st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
          else:
              with st.spinner(text='In progress'):
                process_image(image_file=my_upload,confidence_threshold=confidence_threshold)
              with st.spinner(text='In progress'):
                  time.sleep(3)
                  st.success('Done')

                  bar = st.progress(50)
                  time.sleep(3)
                  bar.progress(100)
                  st.success('Success message')
    # else:
    #   st.write("In else")
    #   st.error('Please upload an image')



