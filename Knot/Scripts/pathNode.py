# # pathNode.py
#
# class Node:
#     def __init__(self, data, point_identifier):
#         self.data = data  # Coordinate to its own point(row, col)
#         self.point_identifier = point_identifier  # "path" or "agent" or "cross" to identify if an agent is needed
#         self.next = None  # Link to the next Node
#
#     def print_node(self):
#         next_point = f" -> {self.next.data} ({self.next.point_identifier})" if self.next else " -> None"
#         print(f"{self.data} ({self.point_identifier}){next_point}")

# pathNode.py

# pathNode.py

from collections import defaultdict


# ---------- Node for Linked Path Representation ----------
class Node:
    def __init__(self, data, point_identifier):
        """
        Node to represent a point on the knot path.
        - data: (row, col) coordinate
        - point_identifier: "path", "agent", or "cross"
        """
        self.data = data
        self.point_identifier = point_identifier
        self.next = None

    def print_node(self):
        next_point = f" -> {self.next.data} ({self.next.point_identifier})" if self.next else " -> None"
        print(f"{self.data} ({self.point_identifier}){next_point}")


# ---------- Knot Point & Segment Structure ----------
class KnotPoint:
    def __init__(self, point_id, row, col, is_agent=False, z=0):
        self.id = point_id
        self.row = row
        self.col = col
        self.z = z
        self.is_agent = is_agent
        self.segments = []  # List of connected KnotSegments

    def pos_2d(self):
        return (self.row, self.col)

    def pos_3d(self):
        return (self.row, self.col, self.z)


class KnotSegment:
    def __init__(self, seg_id, start_point, end_point, over_under=0):
        """
        Segment connects two KnotPoints.
        - over_under: 1 = overpass, -1 = underpass, 0 = flat
        """
        self.id = seg_id
        self.start = start_point
        self.end = end_point
        self.over_under = over_under
        self.crosses = set()
        self.gap_at = []


class KnotGraph:
    def __init__(self):
        self.points = {}   # point_id → KnotPoint
        self.segments = {} # seg_id → KnotSegment
        self.next_point_id = 0
        self.next_segment_id = 0

    def add_point(self, row, col, is_agent=False, z=0):
        point = KnotPoint(self.next_point_id, row, col, is_agent, z)
        self.points[point.id] = point
        self.next_point_id += 1
        return point

    def add_segment(self, start_point, end_point, over_under=0):
        segment = KnotSegment(self.next_segment_id, start_point, end_point, over_under)
        self.segments[segment.id] = segment
        start_point.segments.append(segment)
        end_point.segments.append(segment)
        self.next_segment_id += 1
        return segment


# ---------- Knot Manager (Encapsulates Graph + Agents + Loops) ----------
class KnotManager:
    def __init__(self):
        self.graph = KnotGraph()
        self.agent_registry = {}  # agent_id → KnotPoint
        self.loop_registry = {}   # loop_id → { 'path': [...], 'color': ... }
        self.next_agent_id = 1
        self.next_loop_id = 1
        self.matrix = []
        self.entry = None
        self.exit = None

    def reset(self):
        self.__init__()

    def register_agent(self, point):
        point.is_agent = True
        agent_id = self.next_agent_id
        self.agent_registry[agent_id] = point
        self.next_agent_id += 1
        return agent_id

    def add_loop(self, loop_path):
        loop_id = self.next_loop_id
        self.loop_registry[loop_id] = {"path": loop_path}
        self.next_loop_id += 1
        return loop_id

    def set_matrix(self, matrix, entry, exit_):
        self.matrix = matrix
        self.entry = entry
        self.exit = exit_

