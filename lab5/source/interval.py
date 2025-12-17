class Interval:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        if hi is None:
            lo, hi = float(lo), float(lo)
        else:
            lo, hi = float(lo), float(hi)
        if lo <= hi:
            self.lo, self.hi = lo, hi
        else:
            self.lo, self.hi = hi, lo

    def __repr__(self):
        return f"[{self.lo:.16g}, {self.hi:.16g}]"

    def mid(self):
        return 0.5 * (self.lo + self.hi)

    def rad(self):
        return 0.5 * (self.hi - self.lo)

    def width(self):
        return self.hi - self.lo

    def contains(self, x):
        return self.lo <= x <= self.hi

    def intersect(self, other):
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            return None
        return Interval(lo, hi)

    # basic ops
    def __add__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __mul__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        a, b, c, d = self.lo, self.hi, other.lo, other.hi
        vals = [a * c, a * d, b * c, b * d]
        return Interval(min(vals), max(vals))

    def __truediv__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        # Деление запрещено, если 0 внутри знаменателя
        if other.lo <= 0.0 <= other.hi:
            raise ZeroDivisionError("Interval division by interval containing 0.")
        return self * Interval(1.0 / other.hi, 1.0 / other.lo)

    def square(self):
        # [a,b]^2
        a, b = self.lo, self.hi
        if a >= 0:
            return Interval(a * a, b * b)
        if b <= 0:
            return Interval(b * b, a * a)
        # crosses zero
        return Interval(0.0, max(a * a, b * b))
