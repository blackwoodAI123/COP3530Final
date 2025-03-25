class Graph:

    
    #adjacency list has {"stock_code": connections}
    def __init__(self, nodes):
        self.adjacency_list = {}
        #Creates graph from nodes
        for i in range(len(nodes)):
            connections = []
            for j in range(len(nodes)):
                if (i != j):
                    connections.append(nodes[j])
            connections.sort(key=lambda val:val[1], reverse=True)
            self.adjacency_list[nodes[i][0]] = connections
        self.nodes_to_weight = {}
        #Creates map of node name to value
        for i in range(len(nodes)):
            self.nodes_to_weight[nodes[i][0]] = nodes[i][1]
    
    #traverses graph
    def traverse(self):
        max_code = ""
        max_val = -1
        #Finds the max value in the graph
        for value in self.adjacency_list.keys():
            if self.nodes_to_weight.get(value) != None:
                if self.nodes_to_weight[value] > max_val:
                    max_val = self.nodes_to_weight[value]
                    max_code = value

        current_node = max_code
        sell_order = []
        itr = 0
        #iteratres through graph
        while True:
            #adds to visited
            sell_order.append(current_node)
            new_node = False
            
            #Checks if current node is visited
            if self.adjacency_list.get(current_node) == None:
                break

            #Since the connections are sorted, finds first not visited node
            for value in self.adjacency_list[current_node]:
                if value[0] not in sell_order:
                    current_node = value[0]
                    new_node = True
                    break
            if new_node != True:
                break
                
            #Check for self loops
            itr+=1
            if itr > 100:
                break
        return sell_order
    
    def get_delta(self):
        return self.nodes_to_weight
            