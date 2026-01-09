class Interval:
    def __init__(self, a, b):
        self.a = float(a)
        self.b = float(b)

    def __repr__(self):
        return f"[{self.a}, {self.b}]"

    def mid(self):
        return (self.a + self.b) / 2
    
    def rad(self):
        return (self.b - self.a) / 2

    # Интервальное сложение
    def __add__(self, other):
        return Interval(self.a + other.a, self.b + other.b)

    # Интервальное вычитание
    def __sub__(self, other):
        return Interval(self.a - other.a, self.b - other.b)

    # Интервальное умножение
    def __mul__(self, other):
        products = [
            self.a * other.a,
            self.a * other.b,
            self.b * other.a,
            self.b * other.b
        ]
        return Interval(min(products), max(products))
    
