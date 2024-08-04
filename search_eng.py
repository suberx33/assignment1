import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")



urls = ['https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg',
        'https://animals.sandiegozoo.org/sites/default/files/2016-08/hero_zebra_animals.jpg',
        'https://images.pexels.com/photos/161559/background-bitter-breakfast-bright-161559.jpeg',
        'https://images.pexels.com/photos/102104/pexels-photo-102104.jpeg',
        'https://masterliveaboards.com/wp-content/uploads/2021/04/sea-turltes-facts-gallery-03.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/b/b2/Hausziege_04.jpg',
        'https://cdn01.justjared.com/wp-content/uploads/headlines/2019/05/keanu-reeves-identifies-as-person-of-color.jpg',
        'https://images.pexels.com/photos/106399/pexels-photo-106399.jpeg',
        'https://images.pexels.com/photos/1875480/pexels-photo-1875480.jpeg',
        'https://images.pexels.com/photos/1001682/pexels-photo-1001682.jpeg',
        'https://images.pexels.com/photos/417173/pexels-photo-417173.jpeg',
        'https://www.ikea.com/sa/en/images/products/ekedalen-extendable-table-oak__0736964_pe740828_s5.jpg',
        'https://www.ikea.com/sa/en/images/products/stefan-chair-brown-black__0727320_pe735593_s5.jpg',
        'https://images.pexels.com/photos/775201/pexels-photo-775201.jpeg',
        'https://images.pexels.com/photos/47367/full-moon-moon-bright-sky-47367.jpeg'
        ]

caps = []

images = []

for url in urls:
    response = requests.get(url, stream=True)
    image = Image.open(response.raw)
    images.append(image)

title = st.text_input("Enter text", "Moon")

if st.button("Generate capting", type="primary"):
    #caption = ["goat"]
    caption = title
    input = processor(text=caption, images=images, return_tensors="pt", padding=True)
    output = model(**input)
    probs = output.logits_per_image.argmax()
    image = images[probs.item()]

    st.image(image = image)