#commented
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
#from Utility.linked_list import LinkedList
#from Utility.stock import Stock
from datetime import datetime
#from Utility.portfolio import Portfolio
#from bs4 import BeautifulSoup
#import requests

#CHARLIEoliver2016

from selenium import webdriver
from selenium.webdriver.common.by import By
#from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
#from webdriver_manager.chrome import ChromeDriverManager
#from selenium.webdriver.support.ui import Select
import time

import base64

from io import BytesIO
"""
:D
Monte Carlo Simulation -> Portfolio Optimization
daily

mew = expected return ((end - start) / start) over year
S = stock price {current valued stock price}
dW = random walk of normal variables where if j is the day use j'th value of list
sigma = volatility (standard deviation of past stock data use entire day)
dt = change in time
dS = change in stock price single value

dS = mew * S dt + sigma * s dW

https://finance.yahoo.com/quote/{StockCode}/
"""
#gets potential stocks from the web
def get_data(url):
    #Finds Chrome
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")  

    # Creates the Driver
    driver = webdriver.Chrome(options=options)

    #Opens URL
    driver.get(url)

    #Wait for Page loading
    time.sleep(1)  

    #Goes to Perfmance Tap
    performance_tab = driver.find_element(By.XPATH, "//button[normalize-space(text())='Performance']")
    performance_tab.click()

    

    #Goes to Perfmance Tap
    time.sleep(1)  

    #Data arr
    all_data = []
    for i in range(10):
        #Gets all the stocks
        data = driver.find_elements(By.TAG_NAME, "tr")
        #Gets the values of the data
        for j in data:
            #Gets the individual stock's data
            columns = j.find_elements(By.TAG_NAME, "td")
            row_data = [col.text for col in columns]
            if len(row_data) != 0:
                all_data.append(row_data)

        #Checks if the popup comes up 
        x_button_popup = driver.find_elements("xpath", '//button[@aria-label="Close"]')
        if x_button_popup:
            x_button_popup[0].click()

        #Clicks to next page
        next_button = driver.find_element(By.XPATH, "//button[span[contains(text(), 'Next')]]")
        next_button.click()
        #Goes to Perfmance Tap
        time.sleep(0.05)
    #closes the driver
    driver.close()
    return all_data

#converts a dictionary to a list
def dict_to_list(dict):
    curr_list = []
    for value in dict:
        curr_list.append(f"{value}: {round(dict[value], 3)} shares")
    return curr_list

#Gets potential stocks and processes them into usable data
def get_potential():

    #gets the data
    url = "https://stockanalysis.com/stocks/screener/"
    data = get_data(url)

    """
    Symbol, Company Name, Market Cap, Stock Price, %Change, Change 1W, Change 1M, Change 6M, Change YTD, Change 1Y, CHange 3y Change 5Y
    """
    #converts data to floats
    value = 0
    for i in data:
        i[3] = float(i[3].replace(",", ""))
        for j in range(4, len(i)):
            i[j] = convert_percent(i[j])
        value += 1

    return data
    
#Converts plot into html image
def convert_plot(plot):
    #Saves figure in buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    #Decodes into ascii
    img_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    html_img = f"<img style = 'max-width = 250px; max-height: 250px' src='data:image/png;base64,{img_data}'/>"
    #closes plot
    plt.close(plot)
    return html_img

#Converts Potential Stock to div
def convert_potential_to_div(stock):
    text = f"<p> {stock[0]}, {stock[1]}, {stock[3]} </p>"
    div = f"<div style = 'font-size: 14px'> {text} </div>"
    return div

#converts percent value to float for potential stocks processing
def convert_percent(string_val):
    string_val = string_val.strip()
    string_val = string_val.replace(",", "")
    try:
        return float(string_val[:len(string_val)-1])
    except:
        return -1

#Creates MonteCarlo Graphs
def create_monte_carlo_graphs(portfolio):
    #Gets the data
    curr_datetime = datetime.now().date()
    #Models the Portfolio
    data = portfolio.model_portfolio()
    if data == -1:
        return []
    #Labels of plots
    plt.xlabel(f"Minutes over {curr_datetime} of {portfolio.get_name()}'s portfolio") 
    plt.ylabel('Price in Dollars')
    
    #Names the lines
    itr = 0
    for line in data:
        plt.plot([i for i in range(len(line))], line, label=f"Simulation {itr}")
        itr+=1

    #Sets the upper and lower bounds
    rounded_lower = round(int(portfolio.recalculate_worth() * 0.75) / 100) * 100
    rounded_upper= round(int(portfolio.recalculate_worth() * 1.25) / 100) * 100

    #Margins and limits
    plt.margins(0)
    plt.xlim(0, 390)
    plt.ylim(rounded_lower, rounded_upper)

    #Sets the ticks
    plt.xticks([i for i in range(0, 390, 30)])
    plt.yticks([i for i in range(rounded_lower, rounded_upper, int((rounded_upper - rounded_lower) / 10))])
    return convert_plot(plt.gcf())

#Creates PieChart
def create_pychart(portfolio):
    #Gets the pie chart information
    information = portfolio.info_pie_chart()
    if information == -1:
        return []
    
    #Creates labels
    labels = [i for i in information.keys()]
    values = []
    #adds the values
    for i in information.values():

        if (int(i[0]) <= 0):
            values.append(1)
        else:
            values.append(int(i[0]))
    #Plots the pie chart

    plt.pie(values, labels=labels)
    return convert_plot(plt.gcf())
