# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 02:00:19 2020

@author: Bex.0
"""
from requests_oauthlib import OAuth1Session
import scrapy

# Keys 
consumerKey = 'CA63D668DAC02078C460B0AAAE6587CA'
consumerSecret = '46425CD74850421BA6F1D589DFF9C808'

# https://developer.trademe.co.nz/api-overview/authentication/
oAuthToken = 'IB922FDA0CBED4A6041D30F1086FAD7F'
oAuthSecret = 'A0617E4B5F09EDC2D91BD06FD061F3E4'

tradeMe = OAuth1Session(consumerKey,consumerSecret,oAuthToken,oAuthSecret)
searchHouses = 'https://api.trademe.co.nz/v1/Search/Property/Residential.json'
returnedPageAll = tradeMe.get(searchHouses)
print(returnedPageAll)
class QuotesSpider(scrapy.Spider):
    name = "quotes"

    #def start_requests(self):
    start_urls = [
            'https://www.trademe.co.nz/Browse/CategoryAttributeSearchResults.aspx?search=1&cid=3399&rptpath=350-&sidebar=0&132=PROPERTY&selected135=&selected136=&134=&135=&29=&122=0&122=0&49=0&49=0&153=Dunedin%2c+otago&searchString=Dunedin%2c+otago'
            
        ]
    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'quotes-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)