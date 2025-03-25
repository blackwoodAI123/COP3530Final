#commented
#Node Class
class Node:
    #Constructor
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None
    #Sets the prev node
    def set_prev(self, node):
        self.prev = node

    #Sets the Post Node
    def set_next(self, node):
        self.next = node
    #Gets the Value of the Node
    def get_value(self):
        return self.value

# Doubly Linked Lsit
class LinkedList:
    #Initializes List with values of an array
    def __init__(self, values_list):
        self.total = 0
        if (len(values_list) <= 0):
            self.head = Node(None)
            self.tail = Node(None)
            self.head.next = self.tail
            self.tail.prev = self.head
            self.length = 0
        else:
            #Creates head/tail
            head = Node(values_list[0])
            tail = Node(values_list[len(values_list) - 1])
            self.head = head
            self.tail = tail
            current = head
            self.length = len(values_list)
            self.total += values_list[0]
            for i in range(1, len(values_list)):
                new_node = Node(values_list[i])
                self.total += values_list[i]
                current.set_next(new_node)
                new_node.set_prev(current)
                current = current.next
            self.tail = current
    #Gets head
    def get_head(self):
        return self.head
    #Gets tail
    def get_tail(self):
        return self.tail
    #Pops from the front of the list
    def pop_front(self):
        self.total -= self.head.get_value()
        self.head = self.head.next
        self.head.prev = None
        self.length -= 1
        self.get_mean()
    #Adds to the end of the list
    def append(self, value):
        new_node = Node(value)
        self.total += value
        self.tail.set_next(new_node)
        new_node.set_prev(self.tail)
        self.tail = self.tail.next
        self.length += 1
    #Gets length of list
    def get_length(self):
        return self.length
    #Gets the mean of the list
    def get_mean(self):
        return self.total / self.length