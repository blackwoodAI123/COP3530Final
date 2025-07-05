#commented
from flask import Flask, render_template, url_for, request, redirect, flash, session
import json
import urllib.parse
import yfinance as yf
from datetime import datetime, timedelta
from webscraper import create_monte_carlo_graphs, create_pychart, get_potential, convert_potential_to_div, dict_to_list

from matplotlib import pyplot as plt

#Adjust Imports for new file location
from Utility.linked_list import LinkedList
from Utility.SandP import SandP
from Utility.stock import Stock
from Utility.portfolio import Portfolio


#Creates the app
app = Flask(__name__, template_folder = "Frontend")
"""
url = http://127.0.0.1:5000
"""

#Creates the main function
@app.route('/', methods = ['Post', 'Get'])
def index():
    #Checks if the input is valid
    if request.method == "POST":

        #Gets the stock names/amounts
        stock_names = request.form.getlist("stock_codes")
        stock_amounts = request.form.getlist("stock_amounts")
        
        #Checks if stocks are valid
        failed_stocks = []
        successful_stocks = []
        portfolio_dictionary = {}
        itr = 0
        #Iterates through stock names
        for stock in stock_names:
            #Check is the stock is an empty string
            if stock == "" or stock_amounts[itr] == "":
                itr+=1
                continue
            #Downloads the data
            current_data = yf.download(stock, start=datetime.today()-timedelta(days = 7), end=datetime.today(), interval="1d")

            #Checks if the data is valid or the value is already in the portfolio
            if len(current_data['Close']) <= 2 or portfolio_dictionary.get(stock) != None:
                failed_stocks.append(stock)
                itr+=1
            else:
                portfolio_dictionary[stock] = float(stock_amounts[itr])
                successful_stocks.append(stock)
                itr+=1
        
        #Checks if there are no valid stocks 
        if len(failed_stocks) == len(stock_names) or len(stock_names) == 0:
            return render_template("interface.html", failed_stocks=failed_stocks, successful_stocks=[], sell_instructions=[], 
                                   plots_list=[], plots_list_after = [], potential_stocks=[], before_arr=[], after_arr=[], zipped_data=[])

        else:
            #Gets the risk level/optimization risk/type of optimization
            risk_level = request.form.get("risk_level")
            optimization_risk = request.form.get("optimization_risk")
            type_optimization = request.form.get("type_optimization")

            #Checks if no risk_level is input
            if risk_level == None:
                risk_level = 1
            else:
                risk_level = int(risk_level)

            #Checks if optimization is not input
            if optimization_risk == None:
                optimization_risk = 0
            else:
                optimization_risk = int(optimization_risk)
            #Checks if type of optimization is not input
            if type_optimization == None:
                type_optimization = 2
            else:
                type_optimization = int(type_optimization)
            
            #Creates the user portfolio
            user_portfolio = Portfolio(portfolio_dictionary, 9)
            #Creates monte carlo data
            monte_carlo_data = create_monte_carlo_graphs(user_portfolio)
            #creates pie chart data
            pychart_data = create_pychart(user_portfolio)

            #Different types of optimization done
            if type_optimization == 0:
                sell_instructions = user_portfolio.balance_minvar(risk_level)
            elif type_optimization == 1 and optimization_risk != 0:
                return render_template("interface.html", failed_stocks=[], successful_stocks=[], sell_instructions=[], 
                                   plots_list=[], plots_list_after = [], potential_stocks=[], before_arr=[], after_arr=[], zipped_data=[])
                sell_instructions = user_portfolio.balance_mvo(risk_level, optimization_risk)
            else:
                sell_instructions = user_portfolio.balance_mvo(risk_level)
            
            #Creates a new portfolio with the new stock values
            new_portfolio_dictionary = user_portfolio.get_stock_dict()
            new_portfolio = Portfolio(user_portfolio.get_stock_dict(), 9, user_portfolio.get_cash())

            #Creates new monte carlo data
            monte_carlo_data_after = create_monte_carlo_graphs(new_portfolio)
            #Creates new pie chart data
            pychart_data_after = create_pychart(user_portfolio)
            #Gets the New Data paired
            plots_list = [monte_carlo_data, pychart_data]
            plots_list_after = [monte_carlo_data_after, pychart_data_after]

            #Creates potential stocks for user to pick
            potential_stocks = get_potential()
            used_potential_stocks = []
            for i in range(len(potential_stocks)):
                if user_portfolio.get_stock_dict().get(potential_stocks[i][0]) == None:
                    used_potential_stocks.append(convert_potential_to_div(potential_stocks[i]))
                if len(used_potential_stocks) > 30:
                    break
                
            #Converts dictionaries to lists for portfolio makeup
            before_arr = dict_to_list(portfolio_dictionary)
            after_arr = dict_to_list(new_portfolio_dictionary)

            #converts stock_amounts to ints
            for itr in range(len(stock_amounts)):
                stock_amounts[itr] = int(stock_amounts[itr])
            #Zips data into pairs
            zipped_data = list(zip(stock_names, stock_amounts))


            return render_template("interface.html", failed_stocks=failed_stocks, successful_stocks=successful_stocks, 
                                   sell_instructions=sell_instructions, plots_list=plots_list, plots_list_after = plots_list_after, 
                                   potential_stocks=used_potential_stocks, before_arr=before_arr, after_arr=after_arr, zipped_data=zipped_data)
    else:
        return render_template("interface.html", failed_stocks=[], successful_stocks=[], sell_instructions=[], 
                                   plots_list=[], plots_list_after = [], potential_stocks=[], before_arr=[], after_arr=[], zipped_data=[])

#Runs the file
if __name__ == "__main__":
    app.secret_key = '11123461986'
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_TYPE"] = "filesystem"

    app.run(debug=True)
