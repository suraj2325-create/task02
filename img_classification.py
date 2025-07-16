import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image
from io import BytesIO

# 1. Load Pre-trained Model
model = MobileNetV2(weights='imagenet')

# 2. Download an Image from the Web
url = 'https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg'  # Golden Retriever
response = requests.get(url)
img = Image.open(BytesIO(response.content)).resize((224, 224))

# 3. Preprocess the Image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 4. Predict
preds = model.predict(x)
decoded = decode_predictions(preds, top=3)[0]

# 5. Print Top Predictions
for i, (imagenet_id, label, confidence) in enumerate(decoded):
    print(f"{i+1}. {label}: {confidence*100:.2f}%")

# 6. Show Image with Prediction
plt.imshow(img)
plt.title(f"Prediction: {decoded[0][1]} ({decoded[0][2]*100:.2f}%)")
plt.axis('off')
plt.show()