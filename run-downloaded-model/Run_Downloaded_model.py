
# coding: utf-8

# In[1]:


import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from matplotlib.pyplot import imshow


# In[7]:


# Load model from disk. Replace /YOUR_PATH/model.h5
model_path = 'MnistClassifier.h5'
model = load_model(model_path, compile=False)

# Load example image from disk
img_path='img_5363.png'
img = np.array(Image.open(img_path))

get_ipython().run_line_magic('matplotlib', 'inline')
showme = Image.open(img_path, 'r')
imshow(np.asarray(showme))


# In[8]:


# Use BGR channel ordering. This is not to be done if using cv2 to load the image.
img = img[:, :, ::-1]


# In[9]:


# Normalize image
img = img / 255


# In[10]:


# Add outer shape 1, for batch size
img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))


# In[11]:


# Run model
model.predict(img)

