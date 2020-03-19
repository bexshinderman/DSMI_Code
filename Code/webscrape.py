# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:57:38 2020

@author: Bex.0
"""

from lxml import html #lxml is a superfast library for parsing documents written in markup language

url = 'http://www.nzherald.co.nz' # our target website
response = requests.get(url) #We generate an HTTP request to the target website and store the HTTP response
dom = html.fromstring(response.text) #We parse the HTML document into a tree data structure (the DOM, or document object model)

# we use XPath (a query language for selecting nodes from XML documents) to select the HTML elements 
# containing the data that we are interested in
headlines = dom.xpath('//article[contains(@class,"story-hero")]//h3//a/text()') #We will learn XPATH in the next class
for h in headlines:
    print(h.strip())