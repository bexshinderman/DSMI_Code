# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:46:41 2020

@author: shind
"""

import requests
response = requests.get('http://example.com')
print(response.request.url)
print("\n",response.request.headers)
list1 = dir(response.request)
print("\n",list1)
print("\n status code:", response.status_code)
print("\n", response.headers)
print("\n", response.encoding)
print("\n", response.headers)
print("\n", response.cookies)
print("\n", response.text)
response = requests.post('http://httpbin.org/post', data = {'OP':'Otago Polytech'})
print("\n response:", response.text)

response = requests.put('http://httpbin.org/put', data = {'key':'value'})
print("\n put response:", response.text)
response = requests.delete('http://httpbin.org/delete')
print("\n delete response:", response.text)
response = requests.head('http://httpbin.org/get')
print("/n request head", response.text)
response = requests.options('http://httpbin.org/get')
print("/n request options", response.text)

payload = {'cats': 'cats', 'dogs': 'dogs'}
r = requests.get('http://httpbin.org/get', params=payload)
print(r.url)

url = 'http://httpbin.org/put'
headersDictionary = {'user-agent': 'I\'m a fake browser'}

r = requests.get(url, headers=headersDictionary)
print(r.request.headers)

payload = {'key1': 'value1', 'key2': 'value2'}

response = requests.post("http://httpbin.org/post", data=payload)
print(r.text)

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

response = requests.get('https://i.ytimg.com/vi/kZw-jgCRPeE/maxresdefault.jpg')

binaryImage = Image.open(BytesIO(response.content))
binaryImage.save('./imageRetrievedFromTheWeb.png')
imgplot = plt.imshow(binaryImage)
plt.axis('off')
plt.show()

