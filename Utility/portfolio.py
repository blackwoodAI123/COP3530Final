#commented
from Utility.stock import Stock
from Utility.graph import Graph
from Utility.maxheap import MaxHeap
import math
import numpy as np

class Portfolio:

    #dict of stock names : number of shares
    def __init__(self, stock_codes, risk_goal, cash = 0, name="Bob"):
        
        self.portfolio = {}
        self.stock_array = []
        self.total_worth = 0
        self.cash = cash
        self.name = name
        #Creates the stock codes : number of shares
        for name in stock_codes:
            stock = Stock(name, 10, shares=stock_codes[name])
            self.portfolio[name] = stock
            self.total_worth += stock.get_worth()
            self.stock_array.append(name)
            
        self.risk_goal = risk_goal

        #Sets the stock weights
        for i in range(len(self.stock_array)): 
            if self.total_worth == 0:
                self.portfolio[self.stock_array[i]].set_portfolio_weight(0)
                continue
            self.portfolio[self.stock_array[i]].set_portfolio_weight(self.portfolio[self.stock_array[i]].get_worth() / self.total_worth)

    #Gets the worth of the portfolio
    def get_worth(self):
        return self.total_worth

    #Calculates covariance
    def calculate_covariance(self, current_head1, current_head2, mean1, mean2):
        covariance = 0
        size = 0
        #Calculates difference from mean
        while current_head1 != None and current_head2 != None:
            covariance += (current_head1.get_value() - mean1) * (current_head2.get_value() - mean2)
            current_head1 = current_head1.next
            current_head2 = current_head2.next
            size+=1
        if size == 0:
            return 0
        return covariance / size
    
    #Computes the Variance
    def compute_variance(self):
        #Computes Sigma
        calculate_sigma = 0
        for i in range(len(self.stock_array)):
            current_weight = self.portfolio[self.stock_array[i]].get_portfolio_weight()
            current_sig = self.portfolio[self.stock_array[i]].get_sigma()
            calculate_sigma += pow(current_sig, 2) * pow(current_weight, 2)

        #Computes covariance
        total_covariance = 0
        for k in range(len(self.stock_array)):
            for l in range(k, len(self.stock_array)):
                #Gets the data nodes
                current_list1 = self.portfolio[self.stock_array[k]].get_data_nodes()
                current_list2 = self.portfolio[self.stock_array[l]].get_data_nodes()
                #Gets the heads
                current_head1 = current_list1.get_head()
                current_head2 = current_list2.get_head()
                #Gets the means
                head1_mean = current_list1.get_mean()
                head2_mean = current_list2.get_mean()
                #Calculates the covariance of the stock
                covariance = self.calculate_covariance(current_head1, current_head2, head1_mean, head2_mean)
                covariance /= min(current_list1.get_length(), current_list2.get_length())
                total_term = 2 * self.portfolio[self.stock_array[l]].get_portfolio_weight() * self.portfolio[self.stock_array[k]].get_portfolio_weight() * covariance
                total_covariance += total_term
        #Gets the total variance
        result = total_covariance + calculate_sigma
        return math.sqrt(result)
    
    #Computes variance multiplier if MVO
    def compute_variance_multiplier(self, stock_arr, target_weights, target_risk):
        calculate_sigma = 0
        for i in range(len(stock_arr)):
            current_weight = self.portfolio[stock_arr[i]].get_portfolio_weight()
            current_sig = self.portfolio[stock_arr[i]].get_sigma()
            calculate_sigma += pow(current_sig, 2) * pow(current_weight, 2)

         #find cov   
        total_covariance = 0
        for k in range(len(stock_arr)):
            for l in range(k, len(stock_arr)):
                current_list1 = self.portfolio[stock_arr[k]].get_data_nodes()
                current_list2 = self.portfolio[stock_arr[l]].get_data_nodes()
                current_head1 = current_list1.get_head()
                current_head2 = current_list2.get_head()
                head1_mean = current_list1.get_mean()
                head2_mean = current_list2.get_mean()

                covariance = self.calculate_covariance(current_head1, current_head2, head1_mean, head2_mean)
                covariance /= min(current_list1.get_length(), current_list2.get_length())
                total_term = 2 * target_weights[l] * target_weights[k] * covariance
                total_covariance += total_term
        #Finds the multiplier needed to convert tangency -> mvo
        result = total_covariance + calculate_sigma
        return (target_risk / 100) / math.sqrt(result)

    #Buys stock
    def buy_stock(self, stock_code, shares_num):
        #Checks if stock is in portfolio
        if self.portfolio.get(stock_code) == None:
            #Creates the stock and buys it
            stock = Stock(stock_code, 10, shares_num)
            
            #if self.cash < stock.get_current_price() * shares_num:
                #return False
            self.cash -= stock.get_worth()
            self.portfolio[stock_code] = stock
            self.stock_array.append(stock_code)
            if shares_num == 0:
                return True
            self.total_worth += stock.get_worth()

        else:
            #Adds to the stock value and buys that 
            curr_stock = self.portfolio.get(stock_code)
            
            if round(self.cash, 2) < round(curr_stock.get_current_price() * shares_num, 2):
                additional_money = curr_stock.get_current_price() * shares_num - self.cash
                curr_stock.add_shares(shares_num)
                self.cash = 0
                self.recalculate_worth()
                return additional_money
            else:
                curr_stock.add_shares(shares_num)
                self.cash -= shares_num * curr_stock.get_current_price()
                self.recalculate_worth()
                return True
    
    #Sell the stock
    def sell_stock(self, stock_code, shares_num):
        #Checks if sthe stock code is in the dict and sells if it is
        if self.portfolio.get(stock_code) != None:
            curr_stock = self.portfolio[stock_code]
            if shares_num > curr_stock.get_number_shares():
                shares_num = curr_stock.get_number_shares()
            self.cash += curr_stock.get_current_price() * shares_num
            curr_stock.sell_shares(shares_num)

            #if round(curr_stock.get_worth(), 2) <= 0.00:
                #self.portfolio.pop(stock_code)
                #self.stock_array.remove(stock_code)
            return True
        else:
            print("Failed Sell")
            return False
    
    """
    if any weight over 10% of portfolio, sell for cash
    """
    #Sorts for selling with mvo
    def sort_for_selling_mvo(self):

        
        """
        Optimal Weights Maximize sharpe ratio:
        Sharpe Ratio = (expected return of portfolio - risk free rate) / sigma
        w^optimal = (weights_unit_vect) dot (vector of expected Returns) - Sharpe Ratio 
        -------------------------------------------------------------------------------
        sqrt((weights_unit_vect) dot covariance matrix)

        Var(x1) = variance
        covariance matrix Var(x1) .... Cov(x_n, x_1)
                          Cov(x_n, x_1) ... Var(x_n)



        Optimal Weights Calculation Vector using excess returns
        w^optimal = cov matrix inverse (matrix multiply) excess returns vector      = results in vector of n x 1
        ---------------------------------------------------
        1^T(Transpose of vector of 1's) dot cov matrix inverse (matrix multiply) excess returns vector  = results in scalar

        """

        sharpe = 0
        weight_array = []
        expected_returns_array = []
        stock_name_array = []
        itr = 0
        #Finds covariance Matrix diagnols
        covariance_matrix = [[0 for i in range(len(self.portfolio))] for j in range(len(self.portfolio))]
        for stock in self.portfolio:
            sharpe += self.portfolio[stock].expected_returns() * self.portfolio[stock].get_portfolio_weight()
            expected_returns_array.append(self.portfolio[stock].expected_returns())
            weight_array.append(self.portfolio[stock].get_portfolio_weight())
            stock_name_array.append(stock)
            covariance_matrix.append(pow(self.portfolio[stock].get_sigma(), 2))
            itr += 1
        #Gets the expected returns and populates the 1's vector
        vertical_expected_returns = []
        vector_ones = []
        for expected in expected_returns_array:
            new_arr = np.array(expected)
            new_arr_vertical = new_arr.reshape(-1, 1)
            vertical_expected_returns.append(new_arr_vertical)
            vector_ones.append(1)
        vector_ones_new = np.array(vector_ones)
        vector_ones_vertical = vector_ones_new.reshape(-1, 1)

        #Creates an np array of covariance matrix
        covariance_matrix_np = np.zeros(shape=(len(stock_name_array), len(stock_name_array)))
        index = 0
        for i in range(len(stock_name_array)):
            current_array = []
            for j in range(len(stock_name_array)):
                if i != j:
                    data_i = self.portfolio[stock_name_array[i]].get_data_nodes()
                    data_j = self.portfolio[stock_name_array[j]].get_data_nodes()
                    covariance_matrix[i][j] = self.calculate_covariance(data_i.get_head(), data_j.get_head(), data_i.get_mean(), data_j.get_mean())
                    current_array.append(covariance_matrix[i][j])
                else:
                    #current_array.append(covariance_matrix[i][i])
                    current_array.append(pow(self.portfolio[stock_name_array[i]].get_sigma(), 2))

            covariance_matrix_np[index] = current_array
            index+=1
        
        #Makes the inverse of the covariance matrix
        covariance_matrix_inv = np.linalg.inv(covariance_matrix_np)
        expected_returns_np = np.array(expected_returns_array)

        cov_mult_excess = np.matmul(covariance_matrix_inv, expected_returns_np)
        denominator = np.dot(cov_mult_excess, vector_ones_vertical)
        
        optimal_weights = np.divide(cov_mult_excess, denominator)
        
        optimal_weights_horizontal = optimal_weights.T

        return stock_name_array, optimal_weights_horizontal

    #Balances mvo/tangency portfolio
    def balance_mvo(self, risk_level_selling, *args):
        #Gets the names/weights optimal
        if len(self.stock_array) == 0:
            return []
        instructions = []
        name_arr, weights = self.sort_for_selling_mvo()
        comparison_dict = {}
        
        #Checks for negative values
        negatives = []
        total_negative = 0
        for i in range(len(weights)):
            if (weights[i] <= 0):
                negatives.append(i)
                total_negative += weights[i]
                weights[i] = 0.0
        
        #Divides to get negative values
        total_negative *= -1
        for j in range(len(weights)):
            if j not in negatives and len(negatives) != 0:
                weights[j] -= (total_negative / len(negatives))
        
        #Checks if tangency or mvo
        if len(args) != 0:
            multiplier = self.compute_variance_multiplier(name_arr, weights, args[0])
            for value in weights:
                weights *= multiplier
            instructions.append(f"Optimizing to Given Risk Level: {args[0]}%")
        else:
            instructions.append("Maximizing Sharpe Ratio")
        

        """
        Check if Weights are greater than Zero and normalizes them if not
        """
        total_weight = 0
        for value in weights:
            total_weight += value
        if total_weight > 1.0:
            for i in range(len(weights)):
                weights[i] /= total_weight

        for itr in range(len(name_arr)):
            comparison_dict[name_arr[itr]] = weights[itr]

        for value in comparison_dict:
            print(f"{value} Optimized Weight: {comparison_dict[value]} Current Weight: {self.portfolio[value].get_portfolio_weight()}")

        #Iterates through name array and sells the stocks
        for itr in range(len(name_arr)):
            current_weight = self.portfolio[name_arr[itr]].get_portfolio_weight()
            current_value = (current_weight - weights[itr]) * self.total_worth
            shares_to_delta = current_value / self.portfolio[name_arr[itr]].get_current_price()
            if round(current_weight, 2) > round(weights[itr], 2):
                current_str = f"Sell: {round(float(abs(shares_to_delta)), 3)} of  {name_arr[itr]} @ {self.portfolio[name_arr[itr]].calculate_price_threshold("sell")[risk_level_selling]}"
                instructions.append(current_str)
                if (self.sell_stock(name_arr[itr], shares_to_delta) == False):
                    instructions.pop()
                    instructions.append("Failed Sell")
        #iterates through name array and buys the stocks
        for itr in range(len(name_arr)):
            current_weight = self.portfolio[name_arr[itr]].get_portfolio_weight()
            current_value = (current_weight - weights[itr]) * self.total_worth
            shares_to_delta = current_value / self.portfolio[name_arr[itr]].get_current_price()
            if round(current_weight, 2) < round(weights[itr], 2):
                shares_to_delta *= -1
                current_str = f"Buy: {round(float(abs(shares_to_delta)), 3)} of {name_arr[itr]} @  {self.portfolio[name_arr[itr]].calculate_price_threshold("buy")[risk_level_selling]}"
                instructions.append(current_str)
                success = self.buy_stock(name_arr[itr], shares_to_delta)
                if (success != True):
                    instructions.append(f"Extra Cash Needed : {success}")

        #prints the remaning cash and risk level when selling
        current_str = f"Remaning Cash: {abs(round(self.cash, 2))}"
        instructions.append(current_str)
        risk_level_str = f"Risk Level: {risk_level_selling}"
        instructions.append(risk_level_str)
        #recalculates the worth of the portfolio
        self.recalculate_worth()
        return instructions

    #Sorts for selling with min var
    def sort_for_selling_minvar(self):
        """
        
        inverse covarince matrix dot 1's vertical
        -----------------------------------------
        1 Transposed dot inverse covariance matrix dot 1's vertical
        """

        sharpe = 0
        weight_array = []
        expected_returns_array = []
        stock_name_array = []
        itr = 0
        #Finds covariance Matrix diagnols
        covariance_matrix = [[0 for i in range(len(self.portfolio))] for j in range(len(self.portfolio))]
        for stock in self.portfolio:
            sharpe += self.portfolio[stock].expected_returns() * self.portfolio[stock].get_portfolio_weight()
            expected_returns_array.append(self.portfolio[stock].expected_returns())
            weight_array.append(self.portfolio[stock].get_portfolio_weight())
            stock_name_array.append(stock)
            covariance_matrix.append(pow(self.portfolio[stock].get_sigma(), 2))
            itr += 1
        
        #Finds expected returns
        vertical_expected_returns = []
        vector_ones = []
        for expected in expected_returns_array:
            new_arr = np.array(expected)
            new_arr_vertical = new_arr.reshape(-1, 1)
            vertical_expected_returns.append(new_arr_vertical)
            vector_ones.append(1)

        vector_ones_row= np.array(vector_ones)
        vector_ones_vertical = vector_ones_row.reshape(-1, 1)
        #Calculates the covariance matrix
        covariance_matrix_np = np.zeros(shape=(len(stock_name_array), len(stock_name_array)))
        index = 0
        for i in range(len(stock_name_array)):
            current_array = []
            for j in range(len(stock_name_array)):
                if i != j:
                    data_i = self.portfolio[stock_name_array[i]].get_data_nodes()
                    data_j = self.portfolio[stock_name_array[j]].get_data_nodes()
                    covariance_matrix[i][j] = self.calculate_covariance(data_i.get_head(), data_j.get_head(), data_i.get_mean(), data_j.get_mean())
                    current_array.append(covariance_matrix[i][j])
                else:
                    #current_array.append(covariance_matrix[i][i])
                    current_array.append(pow(self.portfolio[stock_name_array[i]].get_sigma(), 2))

            covariance_matrix_np[index] = current_array
            index+=1
        
        #Gets the inverse of the covairnace matrix
        covariance_matrix_inv = np.linalg.inv(covariance_matrix_np)
        #expected_returns_np = np.array(expected_returns_array)

        #Does the matrix multiplcation to get the optimal weights
        numerator = np.matmul(covariance_matrix_inv, vector_ones_vertical)
        ones_mult = np.matmul(vector_ones_row, covariance_matrix_inv)
        denominator = np.matmul(ones_mult, vector_ones_vertical)

        #cov_mult_excess = np.matmul(covariance_matrix_inv, expected_returns_np)
        #denominator = np.dot(cov_mult_excess, vector_ones_vertical)
        
        optimal_weights = np.divide(numerator, denominator)
        
        optimal_weights_horizontal = optimal_weights.T
        return stock_name_array, optimal_weights_horizontal[0]
    
    #Balances for Minimum Variance
    def balance_minvar(self, risk_level_selling):
        #Gets the weights
        if len(self.stock_array) == 0:
            return []
        instructions = []
        name_arr, weights = self.sort_for_selling_minvar()
        comparison_dict = {}

        #Checks for negative values
        negatives = []
        total_negative = 0
        for i in range(len(weights)):
            if (weights[i] <= 0):
                negatives.append(i)
                total_negative += weights[i]
                weights[i] = 0.0
        
        total_negative *= -1
        for j in range(len(weights)):
            if j not in negatives and len(negatives) != 0:
                weights[j] -= (total_negative / (len(weights) - len(negatives)))
        
        instructions.append("Minimizing Variance")
        
        """
        Check if Weights are greater than Zero
        """
        total_weight = 0
        for value in weights:
            total_weight += value
        if total_weight > 1.0:
            for i in range(len(weights)):
                weights[i] /= total_weight

        
        for i in range(len(weights)):
            comparison_dict[name_arr[i]] = weights[i]
        for value in comparison_dict:
            print(f"{value} Optimized Weight: {comparison_dict[value]} Current Weight: {self.portfolio[value].get_portfolio_weight()}")

        

        sell_stock_arr = []
        buy_stock_arr = []
        for itr in range(len(name_arr)):
            current_weight = self.portfolio[name_arr[itr]].get_portfolio_weight()
            current_value = (current_weight - weights[itr]) * self.total_worth
            shares_to_delta = current_value / self.portfolio[name_arr[itr]].get_current_price()
            if shares_to_delta < 0:
                buy_stock_arr.append([name_arr[itr], -1 * shares_to_delta * self.portfolio[name_arr[itr]].get_current_price(), self.portfolio[name_arr[itr]]])
            else:
                sell_stock_arr.append([name_arr[itr], shares_to_delta * self.portfolio[name_arr[itr]].get_current_price(), self.portfolio[name_arr[itr]]])
        
        """
        Create Sell instructions with max heap
        
        buyHeap = MaxHeap(buy_stock_arr)
        sellHeap = MaxHeap(sell_stock_arr)
        arr_heap_buy = buyHeap.traverse()
        arr_heap_sell = sellHeap.traverse()
        sell_dict = sellHeap.get_delta()
        buy_dict = buyHeap.get_delta()
        for i in arr_heap_buy:
            instructions.append(f"Buy {round(buy_dict[i[0]] / self.portfolio[i[0]].calculate_price_threshold("buy")[risk_level_selling], 3)} shares of {i[0]} @ {self.portfolio[i[0]].calculate_price_threshold("buy")[risk_level_selling]}")
        for i in arr_heap_sell:
            instructions.append(f"Sell {round(sell_dict[i[0]] / self.portfolio[i[0]].calculate_price_threshold("buy")[risk_level_selling], 3)} shares of {i[0]} @ {self.portfolio[i[0]].calculate_price_threshold("buy")[risk_level_selling]}")
        
        """

        """
        Create Sell instructions with Graph
        buyGraph = Graph(buy_stock_arr)
        sellGraph = Graph(sell_stock_arr)
        arr_graph_buy = buyGraph.traverse()
        arr_graph_sell = sellGraph.traverse()
        sell_dict = buyGraph.get_delta()
        buy_dict = buyGraph.get_delta()
    
        for i in arr_graph_sell:
            instructions.append(f"Sell {round(sell_dict[i] / self.portfolio[i].calculate_price_threshold("buy")[risk_level_selling], 3)} shares of {i} @ {self.portfolio[i].calculate_price_threshold("buy")[risk_level_selling]}")
        for i in arr_graph_buy:
            instructions.append(f"Buy {round(buy_dict[i] / self.portfolio[i].calculate_price_threshold("buy")[risk_level_selling], 3)} shares of {i} @ {self.portfolio[i].calculate_price_threshold("buy")[risk_level_selling]}")
        """

        #Sells stocks
        for itr in range(len(name_arr)):
            current_weight = self.portfolio[name_arr[itr]].get_portfolio_weight()
            current_value = (current_weight - weights[itr]) * self.total_worth
            shares_to_delta = current_value / self.portfolio[name_arr[itr]].get_current_price()
            if round(current_weight, 2) > round(weights[itr], 2):
                current_str = f"Sell: {round(float(abs(shares_to_delta)), 3)} of  {name_arr[itr]} @ {self.portfolio[name_arr[itr]].calculate_price_threshold("sell")[risk_level_selling]}"
                instructions.append(current_str)
                if (self.sell_stock(name_arr[itr], shares_to_delta) == False):
                    instructions.pop()
                    instructions.append("Failed Sell")
        #Buys stocks
        for itr in range(len(name_arr)):
            current_weight = self.portfolio[name_arr[itr]].get_portfolio_weight()
            current_value = (current_weight - weights[itr]) * self.total_worth
            shares_to_delta = current_value / self.portfolio[name_arr[itr]].get_current_price()
            if round(current_weight, 2) < round(weights[itr], 2):
                shares_to_delta *= -1
                current_str = f"Buy: {round(float(abs(shares_to_delta)), 3)} of {name_arr[itr]} @  {self.portfolio[name_arr[itr]].calculate_price_threshold("buy")[risk_level_selling]}"
                instructions.append(current_str)
                success = self.buy_stock(name_arr[itr], shares_to_delta)
                if (success != True):
                    instructions.append(f"Extra Cash Needed: {round(success, 2)}")
        
        """
        Finds the variance/Remaining cash
        """
        current_str = f"Remaning Cash: {abs(round(self.cash, 2))}"
        instructions.append(current_str)
        risk_level_str = f"Variance: {round(self.compute_variance() * 100, 2)}%"
        instructions.append(risk_level_str)
        self.recalculate_worth()
        return instructions

    #models the portfolio
    def model_portfolio(self):
        if len(self.stock_array) == 0:
            return -1
        #Calculates the matrix of the first stock
        data = self.portfolio[self.stock_array[0]].calculate_matrix()

        number_shares = self.portfolio[self.stock_array[0]].get_number_shares()
        #Goes through all the stocks and calculates the worth of that stock relative to the portfolio
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] *= number_shares
                data[i][j] += self.cash
        for i in range(1, len(self.stock_array)):
            current_sim = self.portfolio[self.stock_array[i]].calculate_matrix()
                
            number_shares = self.portfolio[self.stock_array[i]].get_number_shares()
            for j in range(len(current_sim)):
                for k in range(len(current_sim[i])):
                    data[j][k] += (current_sim[j][k] * number_shares)
        return data
                    
    # return of 
    # formula of variance sigma1_n of k (weightk^2 * sigmak^2) +  2 * sigma1_n of k weightk * weightk+1 * Cov(k, k+1)
    # sqrt variance to get risk
    #gets the name
    def get_name(self):
        return self.name
    
    #Calculates total worth of portfolio and stock weight
    def recalculate_worth(self):
        
        total_worth = self.cash
        for name in self.stock_array:
           self.portfolio[name].update_current_price()
           total_worth += self.portfolio[name].get_worth()
        self.total_worth = total_worth

        for i in range(len(self.stock_array)):
           self.portfolio[name].set_portfolio_weight(self.portfolio[self.stock_array[i]].get_worth() / self.total_worth)
        return self.total_worth
    
    #Returns the important values for pie chart
    def info_pie_chart(self):
        if len(self.stock_array) == 0:
            return -1
        monies = {}
        for i in range(len(self.stock_array)):
            if float(self.portfolio[self.stock_array[i]].get_worth()) == 0 and float(self.portfolio[self.stock_array[i]].get_portfolio_weight()) == 0:
                continue
            monies[self.stock_array[i]] = [float(self.portfolio[self.stock_array[i]].get_worth()), float(self.portfolio[self.stock_array[i]].get_portfolio_weight())]
        return monies
    
    #Gets the cash
    def get_cash(self):
        return self.cash
    
    #Gets the stock dict of stock name : shares
    def get_stock_dict(self):
        new_dict = {}
        for value in self.stock_array:
            new_dict[value] = self.portfolio[value].get_number_shares()
        return new_dict
