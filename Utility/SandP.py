#commented
from Utility.linked_list import LinkedList
import math
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
#import time

class SandP:
    #Constructor to Set up the Stock
    def __init__(self):
        self.stock_name = "SPY"
        self.data = LinkedList(self.get_info())
        self.last_download = datetime.now()
        self.sigma = self.calculate_sigma()
        pass
    #Gets the information of Stock Prices
    def get_info(self):
        #gets stock prices data
        current_date = datetime.now()
        start_date = current_date - timedelta(days=7)
        data = yf.download(self.stock_name, start=start_date, end=current_date, interval='1m')
        data_array = []
        for value in data['Close'].values:
            data_array.append(float(value))

        return data_array
    
    #Calculates Sigma, i.e. standard deviation or sqrt(variance)
    def calculate_sigma(self):
        total = 0
        curr_node = self.data.head
        next_node = self.data.head.next
        data_points = 0

        while (next_node != None):
            total += next_node.get_value() - curr_node.get_value()
            data_points += 1
            curr_node = curr_node.next
            next_node = next_node.next

        if data_points == 0:
            return 0
        mean = total / data_points
        curr_node = self.data.head
        next_node = self.data.head.next
        summation = 0
        while (next_node != None):
            summation += pow(abs(math.log(next_node.get_value() / curr_node.get_value()) - mean), 2)
            curr_node = curr_node.next
            next_node = next_node.next
        
        summation /= data_points
        return math.sqrt(summation)

    #Updates information to the linked list via new data download
    def update_info(self):
        current_date = datetime.now()
        delta_t = current_date - self.last_download
    
        if delta_t.seconds < 60:
            return
        download = yf.download(self.stock_name, start=self.last_download, end=current_date, interval='1m')
        for value in download['Close'].values:
            self.data.append(float(value))
            self.data.pop_front()
        
        self.last_download = datetime.now()

    #Returns the data in linked list form
    def get_dataLL(self):
        return self.data
    #Gets sigma, i.e. standard deviation or sqrt(variance)
    def get_sigma(self):
        return self.sigma

    #Calculates expected return of SPY
    def calculate_expected_return(self):
        total = 0
        current_head = self.data.get_head()

        while current_head.next != None:
            total += current_head.next.get_value() - current_head.get_value()
            current_head = current_head.next

        return total / self.data.get_length()
