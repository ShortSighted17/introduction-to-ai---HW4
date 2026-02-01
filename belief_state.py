from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, FrozenSet, Optional, List, Set
from parser import GraphData, Edge


class EdgeStatus(Enum):
    UNKNOWN = 'unknown'
    FLOODED = 'flooded'
    CLEAR = 'clear'


@dataclass(frozen=True)
class BeliefState:
    location: int
    has_kit_equipped: bool
    edge_status: Tuple[Tuple[int, EdgeStatus], ...]  # Sorted by edge_id for consistency
    
    @staticmethod
    def create(location: int, has_kit_equipped: bool, 
               edge_statuses: Dict[int, EdgeStatus]) -> 'BeliefState':
        # Sort edge statuses by edge ID for consistent representation
        sorted_statuses = tuple(sorted(edge_statuses.items()))
        return BeliefState(location, has_kit_equipped, sorted_statuses)
    
    def get_edge_status(self, edge_id: int) -> Optional[EdgeStatus]:
        for eid, status in self.edge_status:
            if eid == edge_id:
                return status
        return None
    
    def get_edge_status_dict(self) -> Dict[int, EdgeStatus]:
        return dict(self.edge_status)
    
    def with_location(self, new_location: int) -> 'BeliefState':
        return BeliefState(new_location, self.has_kit_equipped, self.edge_status)
    
    def with_kit_status(self, has_kit: bool) -> 'BeliefState':
        return BeliefState(self.location, has_kit, self.edge_status)
    
    def with_edge_revealed(self, edge_id: int, status: EdgeStatus) -> 'BeliefState':
        new_statuses = dict(self.edge_status)
        if edge_id in new_statuses:
            new_statuses[edge_id] = status
        return BeliefState.create(self.location, self.has_kit_equipped, new_statuses)
    
    def get_unknown_edges(self) -> List[int]:
        return [eid for eid, status in self.edge_status if status == EdgeStatus.UNKNOWN]
    
    def get_known_flooded_edges(self) -> List[int]:
        return [eid for eid, status in self.edge_status if status == EdgeStatus.FLOODED]
    
    def get_known_clear_edges(self) -> List[int]:
        return [eid for eid, status in self.edge_status if status == EdgeStatus.CLEAR]
    
    def is_fully_known(self) -> bool:
        return len(self.get_unknown_edges()) == 0
    
    def __repr__(self):
        status_str = ', '.join(f"E{eid}:{s.value}" for eid, s in self.edge_status)
        kit_str = "with_kit" if self.has_kit_equipped else "no_kit"
        return f"State(V{self.location}, {kit_str}, [{status_str}])"


class BeliefStateSpace:
    def __init__(self, graph: GraphData):
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
        # Start with all uncertain edges unknown
        initial_statuses = {eid: EdgeStatus.UNKNOWN for eid in self.uncertain_edges}
        
        # Create base initial state
        initial = BeliefState.create(
            location=self.graph.start_vertex,
            has_kit_equipped=False,
            edge_statuses=initial_statuses
        )
        return initial
    
    def get_observable_edges_at_location(self, vertex: int) -> Set[int]:
        return self.visible_edges_from.get(vertex, set())
    
    def enumerate_all_states(self) -> List[BeliefState]:
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
        statuses = state.get_edge_status_dict()
        visible = self.get_observable_edges_at_location(state.location)
        
        for eid in visible:
            if statuses.get(eid) == EdgeStatus.UNKNOWN:
                is_flooded = actual_flooding.get(eid, False)
                statuses[eid] = EdgeStatus.FLOODED if is_flooded else EdgeStatus.CLEAR
        
        return BeliefState.create(state.location, state.has_kit_equipped, statuses)
    
    def count_states(self) -> int:
        return len(self.enumerate_all_states())
    
    def print_state_space_info(self):
        print(f"\nBelief State Space Information:")
        print(f"  Vertices: {self.graph.num_vertices}")
        print(f"  Uncertain edges: {len(self.uncertain_edges)} - {self.uncertain_edges}")
        print(f"  Kit equipped options: 2 (True/False)")
        print(f"  Total belief states: {self.count_states()}")
        
        print(f"\n  Edges visible from each vertex:")
        for v in self.graph.get_vertices():
            visible = self.visible_edges_from.get(v, set())
            print(f"    V{v}: {sorted(visible) if visible else 'none'}")
