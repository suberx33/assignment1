import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_files = st.file_uploader(label="Choose an image files",
                 type=['png', 'jpg', 'jpeg'],
                 accept_multiple_files=False)
if img_files is not None:
    image = Image.open(img_files).convert('RGB')

    st.image(image = image,)# caption="Sunrise by the mountains")


    if st.button("Generate capting", type="primary"):
        text = "a photography of"
        inputs = processor(image, text, return_tensors="pt")

        out = model.generate(**inputs)
        # print(processor.decode(out[0], skip_special_tokens=True))

        inputs = processor(image, return_tensors="pt")

        out = model.generate(**inputs)
        capting = processor.decode(out[0], skip_special_tokens=True)
        st.write(capting)