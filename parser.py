"""
Parser for Assignment 4: Hurricane Evacuation Decision Problem (Belief-State MDP)

This module parses input files that define the graph structure for the 
Hurricane Evacuation problem with uncertainty about edge flooding.

INPUT FILE FORMAT:
==================
    #V 5                  ; Number of vertices (1 to n)
    #E1 1 2 W3            ; Edge 1: vertex 1 to 2, weight 3, no flooding possible
    #E2 2 3 W2 F 0.2      ; Edge 2: vertex 2 to 3, weight 2, flooding prob 0.2
    #E3 3 4 W3 F 0.3      ; Edge 3: vertex 3 to 4, weight 3, flooding prob 0.3
    #E4 4 5 W1            ; Edge 4: vertex 4 to 5, weight 1, no flooding
    #E5 2 4 W4            ; Edge 5: vertex 2 to 4, weight 4, no flooding
    
    #K1 1                 ; Amphibian kit located at vertex 1
    #EC 2                 ; Equip cost: 2 time units to equip a kit
    #UC 1                 ; Unequip cost: 1 time unit to unequip
    #FF 3                 ; Flood factor: movement is 3x slower with kit equipped
    
    #O V1 E3 2            ; Observation: from vertex 1, can observe edge 3 status for cost 2
    
    #Start 1              ; Starting vertex
    #Target 5             ; Goal vertex (people to save)

OBSERVATIONS (Additional Requirement):
======================================
The observation directive "#O V<v> E<e> <cost>" specifies that when the agent
is at vertex v, it can pay <cost> time units to observe whether edge e is 
flooded or not. This adds a new action type to the MDP.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set


@dataclass
class Edge:
    """
    Represents an edge in the graph.
    
    Attributes:
        edge_id: Unique identifier for the edge (e.g., 1, 2, 3...)
        u: First endpoint vertex
        v: Second endpoint vertex
        weight: Traversal time (cost) when not using amphibian kit
        flood_prob: Probability that this edge is flooded (0 means never flooded)
    """
    edge_id: int
    u: int
    v: int
    weight: int
    flood_prob: float  # P(flooded), independent of other edges
    
    def is_possibly_flooded(self) -> bool:
        """
        Returns True if this edge has a non-zero probability of being flooded.
        Edges with flood_prob > 0 are the ones we need to track in belief states.
        """
        return self.flood_prob > 0
    
    def get_other_endpoint(self, vertex: int) -> int:
        """
        Given one endpoint, return the other endpoint of this edge.
        Useful for graph traversal.
        """
        if vertex == self.u:
            return self.v
        elif vertex == self.v:
            return self.u
        else:
            raise ValueError(f"Vertex {vertex} is not an endpoint of edge {self.edge_id}")


@dataclass
class Observation:
    """
    Represents an observation action that can be taken at a specific vertex.
    
    This is the additional requirement: from vertex v, the agent can pay a cost
    to observe the flooding status of edge e without traversing to it.
    
    Attributes:
        from_vertex: The vertex where this observation can be made
        edge_id: The edge whose flooding status can be observed
        cost: Time units required to perform this observation
    """
    from_vertex: int
    edge_id: int
    cost: int


@dataclass 
class GraphData:
    """
    Contains all parsed data from the input file.
    
    This data structure holds everything needed to construct the belief-state MDP:
    - Graph topology (vertices and edges)
    - Edge flooding probabilities
    - Amphibian kit locations and usage parameters
    - Observation capabilities (additional requirement)
    - Start and target vertices
    
    Attributes:
        num_vertices: Total number of vertices (numbered 1 to num_vertices)
        edges: Dictionary mapping edge_id to Edge objects
        vertex_edges: Maps each vertex to list of incident edge IDs
        kit_locations: Set of vertices where amphibian kits are located
        equip_cost: Time units to equip an amphibian kit
        unequip_cost: Time units to unequip an amphibian kit
        flood_factor: Movement slowdown factor when kit is equipped
        observations: List of possible observation actions
        start_vertex: Agent's starting position
        target_vertex: Goal vertex where people need to be saved
    """
    num_vertices: int = 0
    edges: Dict[int, Edge] = field(default_factory=dict)
    vertex_edges: Dict[int, List[int]] = field(default_factory=dict)
    
    # Amphibian kit parameters
    kit_locations: Set[int] = field(default_factory=set)
    equip_cost: int = 2      # Default: 2 time units to equip
    unequip_cost: int = 1    # Default: 1 time unit to unequip
    flood_factor: int = 3    # Default: 3x slower with kit
    
    # Observation capabilities (additional requirement)
    observations: List[Observation] = field(default_factory=list)
    
    # Start and target
    start_vertex: Optional[int] = None
    target_vertex: Optional[int] = None
    
    def get_vertices(self) -> List[int]:
        """Return list of all vertex IDs (1 to n)."""
        return list(range(1, self.num_vertices + 1))
    
    def get_incident_edges(self, vertex: int) -> List[Edge]:
        """Return all Edge objects incident to a given vertex."""
        edge_ids = self.vertex_edges.get(vertex, [])
        return [self.edges[eid] for eid in edge_ids]
    
    def get_incident_edge_ids(self, vertex: int) -> List[int]:
        """Return all edge IDs incident to a given vertex."""
        return self.vertex_edges.get(vertex, [])
    
    def get_neighbors(self, vertex: int) -> List[Tuple[int, Edge]]:
        """
        Get all neighbors of a vertex along with the connecting edge.
        
        Returns:
            List of (neighbor_vertex, edge) tuples
        """
        neighbors = []
        for edge in self.get_incident_edges(vertex):
            neighbor = edge.get_other_endpoint(vertex)
            neighbors.append((neighbor, edge))
        return neighbors
    
    def get_possibly_flooded_edges(self) -> List[int]:
        """
        Return IDs of edges that have non-zero flooding probability.
        These are the edges whose status we need to track in belief states.
        """
        return [eid for eid, edge in self.edges.items() if edge.is_possibly_flooded()]
    
    def get_observations_at_vertex(self, vertex: int) -> List[Observation]:
        """
        Return all observation actions available at a given vertex.
        """
        return [obs for obs in self.observations if obs.from_vertex == vertex]
    
    def has_kit_at_vertex(self, vertex: int) -> bool:
        """Check if there's an amphibian kit at this vertex."""
        return vertex in self.kit_locations
    
    def get_traversal_cost(self, edge: Edge, has_kit_equipped: bool) -> int:
        """
        Calculate the time cost to traverse an edge.
        
        If the agent has an amphibian kit equipped, movement is slower
        by the flood_factor multiplier (applies to ALL edges, flooded or not).
        """
        if has_kit_equipped:
            return edge.weight * self.flood_factor
        else:
            return edge.weight


def parse_file(file_path: str) -> GraphData:
    """
    Parse an input file and return the graph data.
    
    The parser handles the following directives:
    - #V n              : Number of vertices
    - #En u v Ww [F p]  : Edge n from u to v, weight w, optional flood prob p
    - #Kn v             : Amphibian kit n at vertex v
    - #EC cost          : Equip cost (time units)
    - #UC cost          : Unequip cost (time units)
    - #FF factor        : Flood factor (slowdown multiplier)
    - #O Vv Ee cost     : Observation from vertex v of edge e for given cost
    - #Start v          : Starting vertex
    - #Target v         : Goal/target vertex
    
    Comments start with semicolon (;) and extend to end of line.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        GraphData object containing all parsed information
    """
    data = GraphData()
    
    with open(file_path, 'r') as f:
        for line in f:
            # Remove comments (everything after semicolon)
            line = line.split(';')[0].strip()
            if not line:
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            # Parse number of vertices: #V 5
            if line.startswith('#V '):
                data.num_vertices = int(parts[1])
                # Initialize vertex_edges for all vertices
                for v in range(1, data.num_vertices + 1):
                    data.vertex_edges[v] = []
            
            # Parse edge: #E1 1 2 W3 [F 0.2]
            elif parts[0].startswith('#E') and parts[0][2:].isdigit():
                edge_id = int(parts[0][2:])  # Extract number after #E
                u = int(parts[1])
                v = int(parts[2])
                
                # Parse weight (format: W followed by number)
                weight = 1  # Default weight
                flood_prob = 0.0  # Default: no flooding possible
                
                for i, tok in enumerate(parts[3:], start=3):
                    if tok.startswith('W'):
                        weight = int(tok[1:])
                    elif tok == 'F':
                        # Next token should be the probability
                        if i + 1 < len(parts):
                            flood_prob = float(parts[i + 1])
                
                edge = Edge(edge_id, u, v, weight, flood_prob)
                data.edges[edge_id] = edge
                
                # Record which edges are incident to which vertices
                if u not in data.vertex_edges:
                    data.vertex_edges[u] = []
                if v not in data.vertex_edges:
                    data.vertex_edges[v] = []
                data.vertex_edges[u].append(edge_id)
                data.vertex_edges[v].append(edge_id)
            
            # Parse amphibian kit location: #K1 1 or #K 1
            elif parts[0].startswith('#K'):
                # Could be #K1 1 or #K 1 - extract the vertex number
                if len(parts[0]) > 2 and parts[0][2:].isdigit():
                    # Format: #K1 1 (kit number 1 at vertex 1)
                    vertex = int(parts[1])
                else:
                    # Format: #K 1 (kit at vertex 1)
                    vertex = int(parts[1])
                data.kit_locations.add(vertex)
            
            # Parse equip cost: #EC 2
            elif parts[0] == '#EC':
                data.equip_cost = int(parts[1])
            
            # Parse unequip cost: #UC 1
            elif parts[0] == '#UC':
                data.unequip_cost = int(parts[1])
            
            # Parse flood factor: #FF 3
            elif parts[0] == '#FF':
                data.flood_factor = int(parts[1])
            
            # Parse observation: #O V1 E3 2 (from vertex 1, observe edge 3, cost 2)
            elif parts[0] == '#O':
                # Format: #O V<vertex> E<edge> <cost>
                from_vertex = int(parts[1][1:])  # Remove 'V' prefix
                edge_id = int(parts[2][1:])      # Remove 'E' prefix
                cost = int(parts[3])
                obs = Observation(from_vertex, edge_id, cost)
                data.observations.append(obs)
            
            # Parse start vertex: #Start 1
            elif parts[0] == '#Start':
                data.start_vertex = int(parts[1])
            
            # Parse target vertex: #Target 5
            elif parts[0] == '#Target':
                data.target_vertex = int(parts[1])
    
    return data


def validate_graph(data: GraphData) -> Tuple[bool, str]:
    """
    Validate the parsed graph data.
    
    Checks:
    1. Start and target vertices are specified
    2. All vertices in edges exist (1 to num_vertices)
    3. All kit locations are valid vertices
    4. All observation actions reference valid vertices and edges
    5. There exists a guaranteed path from start to target or to a kit
       (using only edges with flood_prob = 0)
    
    Returns:
        (is_valid, error_message) tuple
    """
    if data.start_vertex is None:
        return False, "No start vertex specified"
    
    if data.target_vertex is None:
        return False, "No target vertex specified"
    
    if data.start_vertex < 1 or data.start_vertex > data.num_vertices:
        return False, f"Start vertex {data.start_vertex} is out of range"
    
    if data.target_vertex < 1 or data.target_vertex > data.num_vertices:
        return False, f"Target vertex {data.target_vertex} is out of range"
    
    # Check all edges reference valid vertices
    for eid, edge in data.edges.items():
        if edge.u < 1 or edge.u > data.num_vertices:
            return False, f"Edge {eid} has invalid vertex {edge.u}"
        if edge.v < 1 or edge.v > data.num_vertices:
            return False, f"Edge {eid} has invalid vertex {edge.v}"
    
    # Check kit locations are valid
    for kit_vertex in data.kit_locations:
        if kit_vertex < 1 or kit_vertex > data.num_vertices:
            return False, f"Kit at invalid vertex {kit_vertex}"
    
    # Check observations reference valid vertices and edges
    for obs in data.observations:
        if obs.from_vertex < 1 or obs.from_vertex > data.num_vertices:
            return False, f"Observation from invalid vertex {obs.from_vertex}"
        if obs.edge_id not in data.edges:
            return False, f"Observation of invalid edge {obs.edge_id}"
    
    # Check there's a guaranteed path (using only non-floodable edges)
    # from start to either target or an amphibian kit
    reachable = find_guaranteed_reachable(data, data.start_vertex)
    
    # Must be able to reach target or a kit with guaranteed path
    can_reach_target = data.target_vertex in reachable
    can_reach_kit = any(kit in reachable for kit in data.kit_locations)
    
    if not (can_reach_target or can_reach_kit):
        return False, ("Illegal scenario: No guaranteed path from start to target "
                      "or to an amphibian kit using only non-floodable edges")
    
    return True, "Graph is valid"


def find_guaranteed_reachable(data: GraphData, start: int) -> Set[int]:
    """
    Find all vertices reachable from start using only edges that cannot flood.
    
    This is used to validate that there's always a way to reach the target
    or an amphibian kit, regardless of which edges happen to be flooded.
    
    Uses BFS over edges with flood_prob = 0.
    
    Args:
        data: The parsed graph data
        start: Starting vertex
        
    Returns:
        Set of reachable vertex IDs
    """
    reachable = {start}
    queue = [start]
    
    while queue:
        current = queue.pop(0)
        for edge in data.get_incident_edges(current):
            # Only use edges that are guaranteed not to flood
            if edge.flood_prob == 0:
                neighbor = edge.get_other_endpoint(current)
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
    
    return reachable


def print_graph_data(data: GraphData):
    """
    Print a formatted summary of the parsed graph data.
    Useful for debugging and verifying that the input was parsed correctly.
    """
    print("=" * 60)
    print("PARSED GRAPH DATA")
    print("=" * 60)
    
    print(f"\nNumber of vertices: {data.num_vertices}")
    print(f"Start vertex: {data.start_vertex}")
    print(f"Target vertex: {data.target_vertex}")
    
    print(f"\nAmphibian kit parameters:")
    print(f"  Kit locations: {sorted(data.kit_locations) if data.kit_locations else 'None'}")
    print(f"  Equip cost: {data.equip_cost} time units")
    print(f"  Unequip cost: {data.unequip_cost} time units")
    print(f"  Flood factor: {data.flood_factor}x slower movement")
    
    print(f"\nEdges:")
    for eid, edge in sorted(data.edges.items()):
        flood_str = f", P(flood)={edge.flood_prob}" if edge.flood_prob > 0 else ""
        print(f"  E{eid}: {edge.u} -- {edge.v}, weight={edge.weight}{flood_str}")
    
    possibly_flooded = data.get_possibly_flooded_edges()
    print(f"\nPossibly flooded edges: {possibly_flooded if possibly_flooded else 'None'}")
    
    print(f"\nVertex-Edge incidence:")
    for v in data.get_vertices():
        edges = data.vertex_edges.get(v, [])
        print(f"  V{v}: incident edges = {edges}")
    
    if data.observations:
        print(f"\nObservation actions (additional requirement):")
        for obs in data.observations:
            print(f"  From V{obs.from_vertex}: observe E{obs.edge_id} for cost {obs.cost}")
    else:
        print(f"\nNo observation actions defined")


# ============================================================
# TEST CODE - runs when this file is executed directly
# ============================================================

if __name__ == "__main__":
    # Create a test input file matching the assignment example
    test_input = """
#V 5    ; number of vertices n in graph (from 1 to n)

#E1 1 2 W3       ; Edge from vertex 1 to vertex 2, weight 3
#E2 2 3 W2 F 0.2 ; Edge from vertex 2 to vertex 3, weight 2, probability of flooding 0.2
#E3 3 4 W3 F 0.3 ; Edge from vertex 3 to vertex 4, weight 3, probability of flooding 0.3
#E4 4 5 W1       ; Edge from vertex 4 to vertex 5, weight 1
#E5 2 4 W4       ; Edge from vertex 2 to vertex 4, weight 4

#K1 1            ; Amphibian kit at vertex 1
#EC 2            ; Takes 2 units of time to equip an amphibian kit
#UC 1            ; and 1 unit of time to unequip
#FF 3            ; Movement with amphibian kit is 3 times as slow as without

; Additional requirement: observation actions
#O V2 E3 1       ; From vertex 2, can observe edge 3 status for cost 1
#O V3 E2 1       ; From vertex 3, can observe edge 2 status for cost 1

#Start 1
#Target 5
"""
    
    # Write test file
    with open("test_input.txt", "w") as f:
        f.write(test_input)
    
    # Parse and display
    print("Parsing test input file...")
    data = parse_file("test_input.txt")
    print_graph_data(data)
    
    # Validate
    is_valid, message = validate_graph(data)
    print(f"\nValidation: {message}")
