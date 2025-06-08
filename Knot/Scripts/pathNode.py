# pathNode.py

class Node:
    def __init__(self, data, point_identifier):
        self.data = data  # Coordinate to its own point(row, col)
        self.point_identifier = point_identifier  # "path" or "agent" or "cross" to identify if an agent is needed
        self.next = None  # Link to the next Node

    def print_node(self):
        next_point = f" -> {self.next.data} ({self.next.point_identifier})" if self.next else " -> None"
        print(f"{self.data} ({self.point_identifier}){next_point}")