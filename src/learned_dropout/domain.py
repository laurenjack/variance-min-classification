class Dataset:
    def __init__(self, x, y, x_val, y_val):
        self.x = x
        self.y = y
        self.x_val = x_val
        self.y_val = y_val

    def __iter__(self):
        # This allows unpacking like: x, y, x_val, y_val = dataset
        return iter((self.x, self.y, self.x_val, self.y_val))