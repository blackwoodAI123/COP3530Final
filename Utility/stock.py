#Commented
from Utility.linked_list import LinkedList
from Utility.SandP import SandP
import math
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
#import time

beta = SandP()
class Stock:

    """
    mew = expected return ((end - start) / start) over year (Done)
    S = stock price {current valued stock price} Can retrieve using api (Done)
    dW = random walk of normal variables where if j is the day use j'th value of list
    sigma = volatility (standard deviation of past stock data use entire day) (Done)
    dt = change in time in minutes (Done)
    dS = predicted change in stock price

    dS = mew * S dt + sigma * s dW
    """

    #Constructor
    def __init__(self, name, iterations, shares = 0, delta_t = "1m"):
        """
        Name = Stock Name
        Delta T = Change in Time auto = 1 Minute
        """
        #Sets constants
        self.e = 2.718
        self.pi = 3.14159
        self.shares = shares
        self.portfolio_weight = 0
        #self.current_price = yf.download(name , start=datetime.today()-timedelta(minutes=1), end=datetime.today(), interval=delta_t)

        #Sets the current_price
        data = yf.download(name , start=datetime.today()-timedelta(days=1), end=datetime.today(), interval='5d')
        if len(data['Close'].values) > 0:
            self.current_price = data['Close'].values[len(data['Close'].values) - 1]
            self.current_price = float(self.current_price[0])
        else:
            self.current_price = 1
        
        #Sets the variables
        self.stock_name = name
        self.delta_t = delta_t
        self.last_download = datetime.now()
        self.past_data = LinkedList(self.get_info())
        self.sigma = self.calculate_sigma()
        self.drift = self.calculate_drift()

        #Sets the data ticker time if custom is inputted
        end = 0
        for i in range(len(delta_t)):
            if delta_t[i] == '1' or delta_t[i] == '2' or delta_t[i] == '3' or delta_t[i] == '4' or delta_t[i] == '5'or delta_t[i] == '6' or delta_t[i] == '7' or delta_t[i] == '8' or delta_t[i] == '9' or delta_t[i] == '0':
                continue
            end = i
            break
        self.delta_t_n = int(delta_t[0:end])
        #Number of simulations
        self.iterations = iterations
        #Creates the random arr for predictions
        self.random_array = np.random.normal(0, 1, (iterations, 390))
        #calculates the matrix of expected values of stock
        self.calculated_matrix = self.calculate_matrix()
    
    #Gets the name of the stock
    def get_name(self):
        return self.stock_name
    
    #Mew
    def calculate_drift(self):
        drift = 0
        current_node = self.past_data.head
        next_node = self.past_data.head.next
        iterator = 0
        #Sums through linked list
        while (next_node != None):
            end = next_node.get_value()
            start = current_node.get_value()
            drift += math.log(end / start)
            next_node = next_node.next
            current_node = current_node.next
            iterator += 1
        if iterator == 0:
            return 0
        #Returns the drift divided by the size of the LL
        return drift / iterator
    
    #Sigma standard deviation
    def calculate_sigma(self):
        """
        sigma = sqrt(1/n * summation1_n ln(st /st-1) - mean)
        """
       
        total = 0
        curr_node = self.past_data.head
        next_node = self.past_data.head.next
        data_points = 0

        while (next_node != None):
            total += next_node.get_value() - curr_node.get_value()
            data_points += 1
            curr_node = curr_node.next
            next_node = next_node.next

        if data_points == 0:
            return 0
        mean = total / data_points
        curr_node = self.past_data.head
        next_node = self.past_data.head.next
        summation = 0
        while (next_node != None):
            summation += pow(abs(math.log(next_node.get_value() / curr_node.get_value()) - mean), 2)
            curr_node = curr_node.next
            next_node = next_node.next
        
        summation /= data_points
        return math.sqrt(summation)

    #Updates the current price
    def update_current_price(self):
        #Downloads the data and gets the new last value
        data = yf.download(self.stock_name, start=datetime.today()-timedelta(days=1), end=datetime.today(), interval='5d')
        if len(data['Close'].values) > 0:
            self.current_price = data['Close'].values[len(data['Close'].values) - 1]
            self.current_price = float(self.current_price[0])
  
        return self.current_price

    #Gets the stock price info and converts it to floats
    def get_info(self):
        #gets stock prices data
        current_date = datetime.today()
        start_date = current_date - timedelta(days=7)
        data = yf.download(self.stock_name, start=start_date, end=current_date, interval='1m')
        data_array = []
        for value in data['Close'].values:
            data_array.append(float(value))

        return data_array

    #Updste the info form the time last downloaded
    def update_info(self):
        current_date = datetime.now()
        delta_t = current_date - self.last_download
    
        if delta_t.seconds < 60:
            return
        data = yf.download(self.stock_name, start=self.last_download, end=current_date, interval='1m')
        #adds the new values in
        for value in data['Close'].values:
            self.past_data.append(float(value))
            self.past_data.pop_front()
        #updates the drift, sigma, and last download variables
        self.last_download = datetime.now()
        self.drift = self.calculate_drift()
        self.sigma = self.calculate_sigma()

    #Gets the standard deviation
    def get_sigma(self):
        return self.sigma

    #Gets the drift value
    def get_drift(self):
        return self.drift
    #Gets the current price
    def get_current_price(self):
        return self.current_price
    #Sets the current weight of the stock
    def set_portfolio_weight(self, percent):
        self.portfolio_weight = percent
    #Gets the portfolio weight of the stock
    def get_portfolio_weight(self):
        return self.portfolio_weight
    
    #Adds to the number of shares
    def add_shares(self, new_shares):
        self.shares += new_shares
    #sells shares
    def sell_shares(self, new_shares):
        self.shares -= new_shares

    #Gets the worth of the stock relative to the portfolio
    def get_worth(self):
        return self.shares * self.current_price
    
    #Gets the number of shares
    def get_number_shares(self):
        return self.shares
    
    #Gets the LL of data
    def get_data_nodes(self):
        return self.past_data
    
    #Calcualtes the matrix
    def calculate_matrix(self):

        matrix_array = []
    
        """
        Updates the info
        """
        self.update_info()


        #Using real time data Problem? Too sensitive to noise for useful data
        """
        open = 570
        close = 960
        current_minute = self.last_download.hour * 60 + self.last_download.minute
        print("Current minute: ", current_minute)
        minutes = 1
        if current_minute >= open and current_minute <= close:
            minutes += current_minute - 570

        current_sim = []
        tail_node = self.past_data.get_tail()
        while minutes != 1:
            tail_node = tail_node.prev
            minutes -= 1
        while tail_node != None:
            current_sim.append(tail_node.get_value())
            tail_node = tail_node.next
            minutes += 1
        """

        #Iterates through random array
        for i in self.random_array:
            #Create the current sim
            current_sim = []
            current_sim.append(self.current_price)
            #Calculates the sim
            for j in range(1, len(i)):
                #Calculates the difference in stock price expected
                dS_t = self.drift * current_sim[j-1] * self.delta_t_n + self.sigma * current_sim[j-1]*i[j]
                current_sim.append(current_sim[j-1] + dS_t)
                
            matrix_array.append(current_sim)

        return matrix_array

    #Calculates beta, relative to S&P 500
    def calculate_beta(self):
        #Gets the list
        current_list1 = beta.get_dataLL()
        current_list2 = self.past_data
        #Gets the current node
        current_node1 = current_list1.get_head()
        current_node2 = current_list2.get_head()
        #Gets the man
        mean1 = current_list1.get_mean()
        mean2 = current_list2.get_mean()

        #Calculates the variance
        summation = 0
        while current_node1 != None and current_node2 != None:
            summation += (current_node1.get_value() - mean1) * (current_node2.get_value() - mean2)
            current_node1 = current_node1.next
            current_node2 = current_node2.next
        
        #Gets the covariance to the S&P 500
        covariance = summation / (min(current_list1.get_length(), current_list2.get_length()) - 1)
        
        #Gets the beta
        return covariance / beta.get_sigma()

    #Calculates expected returns
    def expected_returns(self):
        """
        Expected return = 
        """
        """
        E(R) = risk free + Beta(Expected of Market  + risk free)
        CAPM model
        """

        #Calculates beta value
        beta_value = self.calculate_beta()
        #Calculates market expected returns
        expectedMarket = beta.calculate_expected_return()

        #Contant rate
        rate = 0.0464
        #Gets risk free return
        risk_free_return = pow((1 + rate), 1/252) - 1
        
        #expected return
        expected_return = risk_free_return + (beta_value * (expectedMarket + risk_free_return))
        return expected_return
    
    #Calculates price threshold
    def calculate_price_threshold(self, operator):
        """
        100% of sims begin at current price
        for selling
            find max(price 75% of sims hit, current price)
            same with 50% and 25%
        for buying
            find min(price 75% of sims hit, current price)
            same with 50% and 25%

        """
        probabilities_array = []
        probabilities_array.append(self.current_price)

        cases = 0
        lower = 0
        middle = 0
        higher = 0
        
        for j in range(len(self.calculated_matrix)):
            current_arr = sorted(self.calculated_matrix[j])
            lower += current_arr[int(len(current_arr) / 4)]
            middle += current_arr[int(len(current_arr) / 2)]
            higher += current_arr[int((3 * len(current_arr)) / 4)]

            cases+=1
        if operator == "sell":
            if cases != 0:
                probabilities_array.append(max(lower / cases, self.current_price))
                probabilities_array.append(max(middle / cases, self.current_price))
                probabilities_array.append(max(higher / cases, self.current_price))
        elif operator == "buy":
            if cases != 0:
                probabilities_array.append(min(higher / cases, self.current_price))
                probabilities_array.append(min(middle / cases, self.current_price))
                probabilities_array.append(min(lower / cases, self.current_price))
        else:
            print("invalid operator")

        if len(probabilities_array) == 0:
            probabilities_array = [0, 0, 0, 0]

        for i in range(4):
            probabilities_array[i] = round(float(probabilities_array[i]), 2)
        return probabilities_array