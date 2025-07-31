class Value:

    def __init__(self, data:float, label:str)->None:
        self.data = data
        self.grad = 0
        self.label = ''
