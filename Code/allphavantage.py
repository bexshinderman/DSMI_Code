# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:34:44 2020

@author: Bex.0
"""
import requests
response = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey=MROQPDQ5LVN9BBSA')
print(response.json())
print(response.request.headers)
print(list(response.json()['Time Series (Daily)'].values())[0]['4. close'])

import matplotlib.pyplot as plt

responseApple = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=AAPL&apikey=YOUR-API-KEY-GOES-HERE')

priceApple = []
for a in list(responseApple.json()['Monthly Time Series'].values()):
    priceApple.append(float(a['4. close']))
plt.plot(priceApple[::-1],c='grey',label='Apple')#We need to reverse the data so x axis goes from past to present 
leg = plt.legend(loc='best')
plt.xlabel("Time in months")
plt.ylabel("Stock price")
plt.show()

responseApple = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=AAPL&apikey=YOUR-API-KEY-GOES-HERE')
responseGoogle = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=GOOG&apikey=YOUR-API-KEY-GOES-HERE')
responseFacebook = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=FB&apikey=YOUR-API-KEY-GOES-HERE')

priceApple = []
priceGoogle = []
priceFacebook = []
for a,g,f in zip(list(responseApple.json()['Monthly Time Series'].values()),
                 list(responseGoogle.json()['Monthly Time Series'].values()),
                 list(responseFacebook.json()['Monthly Time Series'].values())):
    priceApple.append(float(a['4. close']))
    priceGoogle.append(float(g['4. close']))
    priceFacebook.append(float(f['4. close']))
    
import matplotlib.pyplot as plt 
plt.plot(priceApple[::-1],c='grey',label='Apple')
plt.plot(priceGoogle[::-1],c='red',label='Google')
plt.plot(priceFacebook[::-1],c='blue',label='Facebook')

leg = plt.legend(loc='best')
plt.xlabel("Time in months")
plt.ylabel("Stock price")
plt.show()