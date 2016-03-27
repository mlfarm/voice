
"""
    Linear Neural Memory
"""
class Entity(object):
    def __init__(self, vec, data, prev=None, next=None, threashold=0.01):
        self.vec = vec
        self.data = data
        self.prev = prev
        self.next = next
        self.threashold = threashold

    def set_prev(self, prev):
        self.prev = prev

    def set_next(self, next):
        self.next = next

    def __call__(self, query):
        """
            Search query
        """

class MemorySpace(object):
    def __init__(self, threashold=0.01):
        self.threashold = threashold

    def search(self, query):
        
