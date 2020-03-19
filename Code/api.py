# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:19:23 2020

@author: Bex.0
"""

import requests
response = requests.get('https://api.fda.gov/drug/event.json?search=patient.drug.openfda.pharm_class_epc:"nonsteroidal+anti-inflammatory+drug"&count=patient.reaction.reactionmeddrapt.exact')
#print(response.json())
print("****************************************************************************************")
response = requests.get('http://api.openweathermap.org/data/2.5/weather?q=Dunedin,nz&appid=f6b6fecf2c4292d8d19d201e57667588&mode=json')
response.json()
#print(response.json())
print("****************************************************************************************")
response = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey=MROQPDQ5LVN9BBSA')
response.json()
#print(response.json())
