class MaxHeap:

    #values are [stock_code, positive_shares_delta * price, stock object]
    def __init__(self, values = []):
        self.values = values

        #creates heap from array
        for i in range(len(self.values) // 2 - 1, -1, -1):
            self.heapify(i)
        self.stock_to_weight = {}
        #maps stock to amount delta
        for value in self.values:
            self.stock_to_weight[value[0]] = value[1]
    
    #heapifies
    def heapify(self, index):
        largest = index
        left = 2 * index + 1
        right = 2 * index + 2
        #Largest value is left
        if left < len(self.values) and self.values[left][1] > self.values[largest][1]:
            largest = left
        #Largest value is right
        if right < len(self.values) and self.values[right][1] > self.values[largest][1]:
            largest = right
        #IF largest value is not current index
        if largest != index:
            #Swap and go down that tree
            temp = self.values[index]
            self.values[index] = self.values[largest]
            self.values[largest] = temp
            self.heapify(largest)

    #Gets the stock : weight map
    def get_delta(self):
        return self.stock_to_weight
    
    #inserts value
    def insert(self, val):
        self.values.append(val)
        child = len(self.values) - 1

        #heapifies UP
        while child != 0:
            parent = (child - 1) // 2
            if self.values[parent][1] < self.values[child][1]:
                temp = self.values[parent]
                self.values[parent] = self.values[child]
                self.values[child] = temp
                child = parent
            else:
                break
    #deletes value, heapifies down
    def delete(self):
        if (len(self.values) == 0):
            return
        
        return_val = self.values[0]
        self.values[0] = self.values[len(self.values) - 1]
        self.values.pop()
        current_node = 0
        #heapifies down
        while current_node < len(self.values):
            #two kids
            if (2 * current_node) + 1 < len(self.values) and (2 * current_node) + 2 < len(self.values):
                if (self.values[(2 * current_node) + 1][1] < self.values[(2 * current_node) + 2][1] and
                    self.values[current_node] < self.values[(2 * current_node) + 2]):
                    #Swaps
                    temp = self.values[current_node]
                    self.values[current_node] = self.values[(2 * current_node) + 2]
                    self.values[(2 * current_node) + 2] = temp
                    current_node = (2 * current_node) + 2
                elif (self.values[current_node] < self.values[(2 * current_node) + 1]):
                    #Swaps
                    temp = self.values[current_node]
                    self.values[current_node] = self.values[(2 * current_node) + 1]
                    self.values[(2 * current_node) + 1] = temp
                    current_node = (2 * current_node) + 1
                else:
                    break
            #one kid
            elif (2 * current_node) + 1 < len(self.values):
                #Checks if the current values weight is less than the child balues weight
                if (self.values[current_node][1] < self.values[(2 * current_node) + 1][1]):
                    #Swaps
                    temp = self.values[current_node]
                    self.values[current_node] = self.values[(2 * current_node) + 1]
                    self.values[(2 * current_node) + 1] = temp
                    current_node = (2 * current_node) + 1
                else:
                    break
            else:
                break
        return return_val
    #traverses the heap
    
    def traverse(self):
        sell_order = []
        while (len(self.values) != 0):
            sell_order.append(self.delete())
        return sell_order