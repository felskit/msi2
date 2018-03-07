class Point:
    def __init__(self, cls, dist):
        self.cls = cls
        self.dist = dist

    def __lt__(self, other):
        return self.dist < other.dist
