# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:00:31 2020

@author: Bex.0
"""
import requests

url = "https://newsapi.org/v1/articles?source=bbc-news&sortBy=top&apiKey=ee3e0cd95af14146ba34e9d27600abc4"
response = requests.get(url)
for article in response.json()['articles']:
    print(article['description'])
    
url = "https://newsapi.org/v1/articles?source=reuters&sortBy=top&apiKey=ee3e0cd95af14146ba34e9d27600abc4"
response = requests.get(url)
for article in response.json()['articles']:
    print(article['description'])
    print(response.json())
    
from PIL import Image
import requests
from io import BytesIO
url = "https://newsapi.org/v1/articles?source=reuters&sortBy=top&apiKey=ee3e0cd95af14146ba34e9d27600abc4"
response = requests.get(url)
imageUrl = response.json()['articles'][0]['urlToImage']
print(imageUrl)

import matplotlib.pyplot as plt

response = requests.get(imageUrl)
binaryImage = Image.open(BytesIO(response.content))
imgplot = plt.imshow(binaryImage)
plt.axis('off')
plt.show()