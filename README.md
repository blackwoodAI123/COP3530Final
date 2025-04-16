# COP3530Final
Portfolio Optimization
Disclaimer: This optimization algorithm considers daily changes and only gets data from the last 7 days making it very succeptable to changes in the market. To make this algorithm more accurate for higher time intervals, you can extend the amount of data downloaded or change the timeframe of simulations to months/years. Mean Variance Optimization is currently not finished/impossible to do with linear programming. 
Also, please don't actually use these values to trade/optimize your own portfolio without using common sense and reason. :D <br>

File Structure <br>

/Portfolio Optimization <br>
|- Frontend/ <br>
|-- base.html <br>
|-- interface.html <br>

|- static/ <br>
|-- css/ <br>
|--- base.css <br>
|--- interface.css <br>
|-- javascript/ <br>
|--- index.js <br>

|- Utility/ <br>
|-- graph.py <br>
|-- linked_list.py <br>
|-- maxheap.py <br>
|-- portfolio.py <br>
|-- SandP.py <br>
|-- stock.py <br>

|- interface.py <br>
|- webscraper.py <br>


To run the project, run the interface.py file. Some dependencies include but are not limited to numpy, Flask, matplotlib, selenium and pandas.
