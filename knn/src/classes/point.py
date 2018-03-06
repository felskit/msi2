class Point:
    def __init__(self, x, y, cls=None, point=None):
        self.x = x
        self.y = y
        self.cls = cls
        if point:
            self.dist = self._get_dist(point)

    def _get_dist(self, point):
        return ((self.x - point.x) ** 2 + (self.y - point.y) ** 2) ** 1/2

    def __lt__(self, other):
        return self.dist < other.dist
