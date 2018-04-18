class Node(object):

    def __init__(self, value, parent):
        self.value = value
        self.children = []
        self.parent = parent
    
    def setValue(self, val):
        self.value = val
        
    def genChildren(self, dictionary):
        if(isinstance(dictionary, dict)):
            self.children.append(dictionary)

    def getParent(self):
        return self.parent