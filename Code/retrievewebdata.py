# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:18:25 2020

@author: shind
"""
# api key for alphavantage MROQPDQ5LVN9BBSA
import requests
response = requests.get('https://api.fda.gov/drug/event.json?search=patient.drug.openfda.pharm_class_epc:"nonsteroidal+anti-inflammatory+drug"&count=patient.reaction.reactionmeddrapt.exact')
#print(response.json())
print('**********************************************')
response = requests.get('http://api.openweathermap.org/data/2.5/weather?q=Dunedin,nz&appid=f6b6fecf2c4292d8d19d201e57667588&mode=json')
#print(response.json())
print('**********************************************')
response = requests.get('http://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey=MROQPDQ5LVN9BBSA')
#print(response.json())
print('**********************************************')
respone = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=5min&outputsize=full&apikey=demo')
#print(response.json())
print('**********************************************')

daily = list(response.json()['Time Series (Daily)'].values())[0]['4. close']
print(daily)

import matplotlib.pyplot as plt

responseApple = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=AAPL&apikey=MROQPDQ5LVN9BBSA')
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
    plt.plot(priceApple[::-1],c='grey',label='Apple')
plt.plot(priceGoogle[::-1],c='red',label='Google')
plt.plot(priceFacebook[::-1],c='blue',label='Facebook')

leg = plt.legend(loc='best')
plt.xlabel("Time in months")
plt.ylabel("Stock price")
plt.show()