"""
Belief State Representation for Hurricane Evacuation MDP

This module defines the belief state representation for the decision problem.

BELIEF STATE CONCEPT:
=====================
In the Hurricane Evacuation problem, the agent doesn't initially know which edges
are flooded. A "belief state" represents:
1. The agent's current physical location (which vertex)
2. Whether the agent has an amphibian kit equipped
3. What the agent knows about each possibly-flooded edge

For each edge that can be flooded, the agent's knowledge can be:
- UNKNOWN: The agent hasn't observed this edge yet
- FLOODED: The agent has observed/learned that this edge IS flooded
- CLEAR: The agent has observed/learned that this edge is NOT flooded

Edge status is revealed when:
1. The agent reaches a vertex adjacent to the edge (automatic observation)
2. The agent uses an observation action (additional requirement)

STATE SPACE SIZE:
=================
For n possibly-flooded edges, each can be in 3 states (unknown/flooded/clear).
Combined with location (V vertices) and kit status (2 states), the total
state space is approximately: V * 2 * 3^n

This is why the assignment limits us to at most 10 possibly flooded edges
(3^10 = 59,049 states per location/kit combination).
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, FrozenSet, Optional, List, Set
from parser import GraphData, Edge


class EdgeStatus(Enum):
    """
    Represents what the agent knows about an edge's flooding status.
    
    UNKNOWN: Agent hasn't observed this edge yet - it might or might not be flooded
    FLOODED: Agent has confirmed this edge IS flooded (cannot traverse without kit)
    CLEAR: Agent has confirmed this edge is NOT flooded (safe to traverse)
    """
    UNKNOWN = 'unknown'
    FLOODED = 'flooded'
    CLEAR = 'clear'


@dataclass(frozen=True)
class BeliefState:
    """
    Represents a belief state in the Hurricane Evacuation MDP.
    
    This is an immutable (frozen) dataclass so it can be used as a dictionary key
    and stored in sets. Each belief state uniquely identifies what the agent
    knows at a given point in time.
    
    Attributes:
        location: Current vertex where the agent is located
        has_kit_equipped: Whether the agent currently has an amphibian kit equipped
        edge_status: Tuple of (edge_id, status) pairs for all possibly-flooded edges
                    Using a tuple (immutable) instead of dict for hashability
    """
    location: int
    has_kit_equipped: bool
    edge_status: Tuple[Tuple[int, EdgeStatus], ...]  # Sorted by edge_id for consistency
    
    @staticmethod
    def create(location: int, has_kit_equipped: bool, 
               edge_statuses: Dict[int, EdgeStatus]) -> 'BeliefState':
        """
        Factory method to create a BeliefState with proper sorting.
        
        This ensures that equivalent belief states are always represented
        the same way (edges sorted by ID), which is crucial for hashing.
        """
        # Sort edge statuses by edge ID for consistent representation
        sorted_statuses = tuple(sorted(edge_statuses.items()))
        return BeliefState(location, has_kit_equipped, sorted_statuses)
    
    def get_edge_status(self, edge_id: int) -> Optional[EdgeStatus]:
        """
        Get the known status of a specific edge.
        
        Returns None if the edge is not tracked (always clear edges).
        """
        for eid, status in self.edge_status:
            if eid == edge_id:
                return status
        return None
    
    def get_edge_status_dict(self) -> Dict[int, EdgeStatus]:
        """Convert edge_status tuple back to a dictionary for easier manipulation."""
        return dict(self.edge_status)
    
    def with_location(self, new_location: int) -> 'BeliefState':
        """Create a new BeliefState with a different location."""
        return BeliefState(new_location, self.has_kit_equipped, self.edge_status)
    
    def with_kit_status(self, has_kit: bool) -> 'BeliefState':
        """Create a new BeliefState with different kit status."""
        return BeliefState(self.location, has_kit, self.edge_status)
    
    def with_edge_revealed(self, edge_id: int, status: EdgeStatus) -> 'BeliefState':
        """
        Create a new BeliefState with an edge's status revealed.
        
        This is used when the agent observes an edge (either by being adjacent
        to it or by using an observation action).
        """
        new_statuses = dict(self.edge_status)
        if edge_id in new_statuses:
            new_statuses[edge_id] = status
        return BeliefState.create(self.location, self.has_kit_equipped, new_statuses)
    
    def get_unknown_edges(self) -> List[int]:
        """Return list of edge IDs whose status is still unknown."""
        return [eid for eid, status in self.edge_status if status == EdgeStatus.UNKNOWN]
    
    def get_known_flooded_edges(self) -> List[int]:
        """Return list of edge IDs known to be flooded."""
        return [eid for eid, status in self.edge_status if status == EdgeStatus.FLOODED]
    
    def get_known_clear_edges(self) -> List[int]:
        """Return list of edge IDs known to be clear."""
        return [eid for eid, status in self.edge_status if status == EdgeStatus.CLEAR]
    
    def is_fully_known(self) -> bool:
        """Check if all possibly-flooded edges have been observed."""
        return len(self.get_unknown_edges()) == 0
    
    def __repr__(self):
        """Human-readable representation for debugging."""
        status_str = ', '.join(f"E{eid}:{s.value}" for eid, s in self.edge_status)
        kit_str = "with_kit" if self.has_kit_equipped else "no_kit"
        return f"State(V{self.location}, {kit_str}, [{status_str}])"


class BeliefStateSpace:
    """
    Manages the enumeration and manipulation of all possible belief states.
    
    This class is responsible for:
    1. Generating all possible belief states for the MDP
    2. Computing initial belief states
    3. Determining which edges are observable from each vertex
    4. Computing state transitions based on actions
    """
    
    def __init__(self, graph: GraphData):
        """
        Initialize the belief state space from the graph data.
        
        Args:
            graph: Parsed graph containing vertices, edges, and parameters
        """
        self.graph = graph
        
        # Get the list of edges that can potentially be flooded
        # These are the only edges we need to track in belief states
        self.uncertain_edges = graph.get_possibly_flooded_edges()
        
        # For efficiency, precompute which edges are visible from each vertex
        # An edge is visible when the agent is at one of its endpoints
        self.visible_edges_from: Dict[int, Set[int]] = {}
        for v in graph.get_vertices():
            self.visible_edges_from[v] = set()
            for edge in graph.get_incident_edges(v):
                if edge.edge_id in self.uncertain_edges:
                    self.visible_edges_from[v].add(edge.edge_id)
        
        # Store all generated belief states
        self._all_states: Optional[List[BeliefState]] = None
    
    def get_initial_belief_state(self) -> BeliefState:
        """
        Create the initial belief state at the start of the problem.
        
        Initially:
        - Agent is at the start vertex
        - No kit equipped (agent must equip it if they want to use it)
        - All possibly-flooded edges are UNKNOWN
        
        However, edges visible from the starting position are immediately revealed.
        """
        # Start with all uncertain edges unknown
        initial_statuses = {eid: EdgeStatus.UNKNOWN for eid in self.uncertain_edges}
        
        # Create base initial state
        initial = BeliefState.create(
            location=self.graph.start_vertex,
            has_kit_equipped=False,
            edge_statuses=initial_statuses
        )
        
        # Note: We don't reveal edges at start - that happens during state enumeration
        # when we consider what information is revealed at each location
        return initial
    
    def get_observable_edges_at_location(self, vertex: int) -> Set[int]:
        """
        Get the set of uncertain edge IDs that become visible at a vertex.
        
        An edge is visible when the agent is at one of its endpoints.
        This is used for automatic observation when the agent moves.
        """
        return self.visible_edges_from.get(vertex, set())
    
    def enumerate_all_states(self) -> List[BeliefState]:
        """
        Enumerate all possible belief states for the MDP.
        
        A belief state consists of:
        1. Location: one of V vertices
        2. Kit equipped: True or False
        3. Edge knowledge: for each uncertain edge, one of {UNKNOWN, FLOODED, CLEAR}
        
        However, not all combinations are valid:
        - If an edge is adjacent to the current location, it MUST be known
          (either FLOODED or CLEAR, not UNKNOWN)
        
        This reduces the state space significantly.
        """
        if self._all_states is not None:
            return self._all_states
        
        all_states = []
        
        # For each vertex
        for vertex in self.graph.get_vertices():
            # For each kit status
            for has_kit in [False, True]:
                # Get which edges MUST be known at this location
                must_know = self.visible_edges_from.get(vertex, set())
                can_be_unknown = [e for e in self.uncertain_edges if e not in must_know]
                
                # Generate all valid combinations of edge knowledge
                # Edges in must_know must be FLOODED or CLEAR
                # Edges in can_be_unknown can be UNKNOWN, FLOODED, or CLEAR
                for edge_config in self._generate_edge_configs(must_know, can_be_unknown):
                    state = BeliefState.create(vertex, has_kit, edge_config)
                    all_states.append(state)
        
        self._all_states = all_states
        return all_states
    
    def _generate_edge_configs(self, must_know: Set[int], 
                                can_be_unknown: List[int]) -> List[Dict[int, EdgeStatus]]:
        """
        Generate all valid edge status configurations.
        
        Args:
            must_know: Set of edge IDs that must be FLOODED or CLEAR
            can_be_unknown: List of edge IDs that can also be UNKNOWN
            
        Returns:
            List of dictionaries mapping edge_id -> EdgeStatus
        """
        configs = []
        
        # Status options for each edge type
        known_options = [EdgeStatus.FLOODED, EdgeStatus.CLEAR]
        unknown_options = [EdgeStatus.UNKNOWN, EdgeStatus.FLOODED, EdgeStatus.CLEAR]
        
        # Sort edges for consistent ordering
        must_know_list = sorted(must_know)
        can_be_unknown_sorted = sorted(can_be_unknown)
        
        # Generate all combinations using recursion
        def generate(must_know_idx: int, unknown_idx: int, 
                    current_config: Dict[int, EdgeStatus]):
            # Base case: all edges assigned
            if must_know_idx == len(must_know_list) and unknown_idx == len(can_be_unknown_sorted):
                configs.append(current_config.copy())
                return
            
            # Assign next must-know edge
            if must_know_idx < len(must_know_list):
                eid = must_know_list[must_know_idx]
                for status in known_options:
                    current_config[eid] = status
                    generate(must_know_idx + 1, unknown_idx, current_config)
            # Assign next can-be-unknown edge
            elif unknown_idx < len(can_be_unknown_sorted):
                eid = can_be_unknown_sorted[unknown_idx]
                for status in unknown_options:
                    current_config[eid] = status
                    generate(must_know_idx, unknown_idx + 1, current_config)
        
        generate(0, 0, {})
        return configs
    
    def reveal_edges_at_location(self, state: BeliefState, 
                                  actual_flooding: Dict[int, bool]) -> BeliefState:
        """
        Update a belief state by revealing edges visible from the current location.
        
        This is called when the agent moves to a new vertex. Any uncertain edges
        adjacent to that vertex are revealed according to the actual flooding status.
        
        Args:
            state: Current belief state
            actual_flooding: Ground truth - which edges are actually flooded
            
        Returns:
            New belief state with revealed edges
        """
        statuses = state.get_edge_status_dict()
        visible = self.get_observable_edges_at_location(state.location)
        
        for eid in visible:
            if statuses.get(eid) == EdgeStatus.UNKNOWN:
                is_flooded = actual_flooding.get(eid, False)
                statuses[eid] = EdgeStatus.FLOODED if is_flooded else EdgeStatus.CLEAR
        
        return BeliefState.create(state.location, state.has_kit_equipped, statuses)
    
    def count_states(self) -> int:
        """Return the total number of belief states."""
        return len(self.enumerate_all_states())
    
    def print_state_space_info(self):
        """Print information about the belief state space."""
        print(f"\nBelief State Space Information:")
        print(f"  Vertices: {self.graph.num_vertices}")
        print(f"  Uncertain edges: {len(self.uncertain_edges)} - {self.uncertain_edges}")
        print(f"  Kit equipped options: 2 (True/False)")
        print(f"  Total belief states: {self.count_states()}")
        
        print(f"\n  Edges visible from each vertex:")
        for v in self.graph.get_vertices():
            visible = self.visible_edges_from.get(v, set())
            print(f"    V{v}: {sorted(visible) if visible else 'none'}")


# ============================================================
# TEST CODE
# ============================================================

if __name__ == "__main__":
    from parser import parse_file, print_graph_data
    
    # Create test input
    test_input = """
#V 4
#E1 1 2 W1 F 0.3
#E2 2 3 W2 F 0.4
#E3 3 4 W1
#E4 1 4 W3
#Start 1
#Target 4
"""
    
    with open("test_belief.txt", "w") as f:
        f.write(test_input)
    
    graph = parse_file("test_belief.txt")
    print_graph_data(graph)
    
    space = BeliefStateSpace(graph)
    space.print_state_space_info()
    
    # Enumerate states
    states = space.enumerate_all_states()
    print(f"\nFirst 10 belief states:")
    for i, state in enumerate(states[:10]):
        print(f"  {i+1}. {state}")
