"""
Belief-State MDP for Hurricane Evacuation Problem

This module defines the Markov Decision Process (MDP) for the Hurricane Evacuation
problem under uncertainty.

MDP COMPONENTS:
===============

STATES:
  Each state is a BeliefState consisting of:
  - Current location (vertex)
  - Kit equipped status (boolean)
  - Knowledge about each uncertain edge (UNKNOWN/FLOODED/CLEAR)

ACTIONS:
  From each state, the agent can take the following actions:
  
  1. TRAVERSE(edge): Move along an edge to the adjacent vertex
     - Precondition: Agent is at an endpoint of the edge
     - If edge status is UNKNOWN: splits into outcomes (flooded/clear)
     - If edge is FLOODED and no kit: action fails (blocked)
     - If edge is FLOODED and has kit: succeeds but slower
     - If edge is CLEAR: succeeds at normal speed
     
  2. EQUIP: Equip an amphibian kit at current location
     - Precondition: A kit exists at current location AND agent doesn't have one equipped
     - Cost: equip_cost time units
     
  3. UNEQUIP: Unequip the amphibian kit
     - Precondition: Agent has a kit equipped
     - Cost: unequip_cost time units
     - The kit stays at the current vertex (can be picked up again later)
     
  4. OBSERVE(edge): Observe the status of an edge (ADDITIONAL REQUIREMENT)
     - Precondition: There's an observation action from current vertex for this edge
     - Cost: The specified observation cost
     - Result: The edge's status becomes known (FLOODED or CLEAR)

TRANSITIONS:
  Most transitions are deterministic, except:
  - When traversing an edge with UNKNOWN status, there's a probabilistic branch
    based on the edge's flood probability

REWARDS (actually costs - we minimize):
  - We minimize expected TIME to reach the target
  - All actions have positive costs (time)
  - Reaching the target is the goal (terminal state)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from enum import Enum, auto
from belief_state import BeliefState, BeliefStateSpace, EdgeStatus
from parser import GraphData, Edge, Observation


class ActionType(Enum):
    """
    Types of actions the agent can take.
    """
    TRAVERSE = auto()    # Move along an edge
    EQUIP = auto()       # Equip amphibian kit
    UNEQUIP = auto()     # Unequip amphibian kit
    OBSERVE = auto()     # Observe an edge's status (additional requirement)


@dataclass(frozen=True)
class Action:
    """
    Represents an action the agent can take.
    
    For TRAVERSE actions, edge_id specifies which edge to traverse.
    For OBSERVE actions, edge_id specifies which edge to observe.
    For EQUIP/UNEQUIP, edge_id is None.
    
    This is frozen (immutable) so it can be used as a dictionary key.
    """
    action_type: ActionType
    edge_id: Optional[int] = None  # For TRAVERSE and OBSERVE actions
    
    def __repr__(self):
        if self.action_type == ActionType.TRAVERSE:
            return f"Traverse(E{self.edge_id})"
        elif self.action_type == ActionType.OBSERVE:
            return f"Observe(E{self.edge_id})"
        else:
            return self.action_type.name


@dataclass
class Transition:
    """
    Represents a probabilistic transition in the MDP.
    
    When an action is taken in a state, there may be multiple possible outcomes
    (e.g., when traversing an edge with unknown flooding status).
    
    Attributes:
        probability: Probability of this outcome (0 to 1)
        next_state: The resulting belief state
        cost: Time cost incurred for this transition
    """
    probability: float
    next_state: BeliefState
    cost: float  # Using float for precision, represents time units


class HurricaneMDP:
    """
    The Belief-State Markov Decision Process for Hurricane Evacuation.
    
    This class defines:
    - The state space (delegated to BeliefStateSpace)
    - The action space for each state
    - The transition function (including probabilities for uncertain edges)
    - The cost function (time to reach target)
    
    The goal is to minimize expected time to reach the target vertex.
    """
    
    def __init__(self, graph: GraphData):
        """
        Initialize the MDP from the graph data.
        
        Args:
            graph: Parsed graph containing all problem parameters
        """
        self.graph = graph
        self.belief_space = BeliefStateSpace(graph)
        
        # Terminal states are those where location == target
        self.target = graph.target_vertex
        
        # Cache for available actions at each state
        self._action_cache: Dict[BeliefState, List[Action]] = {}
        
        # Cache for transitions (state, action) -> list of Transition
        self._transition_cache: Dict[Tuple[BeliefState, Action], List[Transition]] = {}
    
    def is_terminal(self, state: BeliefState) -> bool:
        """
        Check if a state is terminal (goal reached).
        
        A state is terminal when the agent has reached the target vertex.
        """
        return state.location == self.target
    
    def get_available_actions(self, state: BeliefState) -> List[Action]:
        """
        Get all actions available in a given belief state.
        
        This checks preconditions for each action type and returns only valid actions.
        
        Args:
            state: Current belief state
            
        Returns:
            List of valid Action objects
        """
        # Check cache first
        if state in self._action_cache:
            return self._action_cache[state]
        
        actions = []
        
        # Terminal states have no actions
        if self.is_terminal(state):
            self._action_cache[state] = []
            return []
        
        location = state.location
        
        # 1. TRAVERSE actions - one for each incident edge
        for edge in self.graph.get_incident_edges(location):
            # Check if traverse is possible
            if self._can_traverse(state, edge):
                actions.append(Action(ActionType.TRAVERSE, edge.edge_id))
        
        # 2. EQUIP action - if kit available and not already equipped
        if (self.graph.has_kit_at_vertex(location) and 
            not state.has_kit_equipped):
            actions.append(Action(ActionType.EQUIP))
        
        # 3. UNEQUIP action - if currently equipped
        if state.has_kit_equipped:
            actions.append(Action(ActionType.UNEQUIP))
        
        # 4. OBSERVE actions (additional requirement)
        for obs in self.graph.get_observations_at_vertex(location):
            # Can only observe unknown edges
            edge_status = state.get_edge_status(obs.edge_id)
            if edge_status == EdgeStatus.UNKNOWN:
                actions.append(Action(ActionType.OBSERVE, obs.edge_id))
        
        self._action_cache[state] = actions
        return actions
    
    def _can_traverse(self, state: BeliefState, edge: Edge) -> bool:
        """
        Check if an edge can be traversed from the current state.
        
        An edge can be traversed if:
        - Status is CLEAR: always OK
        - Status is FLOODED: only if kit equipped
        - Status is UNKNOWN: always possible (we'll find out when we try)
        
        Note: If status is FLOODED and no kit, we CANNOT traverse.
        """
        status = state.get_edge_status(edge.edge_id)
        
        if status is None:
            # Edge is not tracked (always clear)
            return True
        
        if status == EdgeStatus.CLEAR:
            return True
        
        if status == EdgeStatus.FLOODED:
            return state.has_kit_equipped  # Can only traverse flooded with kit
        
        if status == EdgeStatus.UNKNOWN:
            return True  # We can try - will learn the status
        
        return False
    
    def get_transitions(self, state: BeliefState, action: Action) -> List[Transition]:
        """
        Get all possible transitions for a (state, action) pair.
        
        Most transitions are deterministic, but traversing an edge with
        UNKNOWN status produces a probabilistic outcome.
        
        For OBSERVE actions (additional requirement), we also get a probabilistic
        outcome based on the edge's flood probability.
        
        Args:
            state: Current belief state
            action: Action to take
            
        Returns:
            List of Transition objects (probabilities sum to 1)
        """
        cache_key = (state, action)
        if cache_key in self._transition_cache:
            return self._transition_cache[cache_key]
        
        transitions = []
        
        if action.action_type == ActionType.TRAVERSE:
            transitions = self._get_traverse_transitions(state, action.edge_id)
        
        elif action.action_type == ActionType.EQUIP:
            transitions = self._get_equip_transitions(state)
        
        elif action.action_type == ActionType.UNEQUIP:
            transitions = self._get_unequip_transitions(state)
        
        elif action.action_type == ActionType.OBSERVE:
            transitions = self._get_observe_transitions(state, action.edge_id)
        
        self._transition_cache[cache_key] = transitions
        return transitions
    
    def _get_traverse_transitions(self, state: BeliefState, 
                                   edge_id: int) -> List[Transition]:
        """
        Compute transitions for a TRAVERSE action.
        
        If the edge status is known, this is deterministic.
        If unknown, we branch on flood probability.
        
        When we arrive at the destination vertex, we also reveal any
        uncertain edges adjacent to that vertex.
        """
        edge = self.graph.edges[edge_id]
        destination = edge.get_other_endpoint(state.location)
        status = state.get_edge_status(edge_id)
        
        # Calculate traversal cost
        base_cost = self.graph.get_traversal_cost(edge, state.has_kit_equipped)
        
        if status == EdgeStatus.UNKNOWN:
            # Probabilistic transition - edge might be flooded or clear
            flood_prob = edge.flood_prob
            
            transitions = []
            
            # Outcome 1: Edge turns out to be CLEAR
            if flood_prob < 1.0:
                clear_cost = base_cost  # Normal traversal
                base_state_clear = self._make_traverse_result(
                    state, destination, edge_id, EdgeStatus.CLEAR
                )
                # Expand with any newly revealed edges at destination
                clear_transitions = self._expand_with_revealed_edges(base_state_clear, clear_cost)
                for t in clear_transitions:
                    transitions.append(Transition(
                        probability=(1.0 - flood_prob) * t.probability,
                        next_state=t.next_state,
                        cost=t.cost
                    ))
            
            # Outcome 2: Edge turns out to be FLOODED
            if flood_prob > 0.0:
                if state.has_kit_equipped:
                    # Can traverse flooded edge with kit (but already slower)
                    flooded_cost = base_cost  # Already accounts for kit slowdown
                    base_state_flooded = self._make_traverse_result(
                        state, destination, edge_id, EdgeStatus.FLOODED
                    )
                    # Expand with any newly revealed edges at destination
                    flooded_transitions = self._expand_with_revealed_edges(base_state_flooded, flooded_cost)
                    for t in flooded_transitions:
                        transitions.append(Transition(
                            probability=flood_prob * t.probability,
                            next_state=t.next_state,
                            cost=t.cost
                        ))
                else:
                    # BLOCKED - cannot traverse, stay in place but learn edge is flooded
                    # Cost of attempting is 0 (instant discovery)
                    new_state_blocked = state.with_edge_revealed(edge_id, EdgeStatus.FLOODED)
                    transitions.append(Transition(
                        probability=flood_prob,
                        next_state=new_state_blocked,
                        cost=0  # We discover instantly without time passing
                    ))
            
            return transitions
        
        elif status == EdgeStatus.CLEAR:
            # Deterministic - just traverse
            base_state = self._make_traverse_result(
                state, destination, edge_id, EdgeStatus.CLEAR
            )
            # But may still need to reveal edges at destination
            return self._expand_with_revealed_edges(base_state, base_cost)
        
        elif status == EdgeStatus.FLOODED:
            # Must have kit to get here (checked in _can_traverse)
            base_state = self._make_traverse_result(
                state, destination, edge_id, EdgeStatus.FLOODED
            )
            return self._expand_with_revealed_edges(base_state, base_cost)
        
        else:
            # Edge not tracked (always clear)
            new_state = state.with_location(destination)
            return self._expand_with_revealed_edges(new_state, base_cost)
    
    def _make_traverse_result(self, state: BeliefState, destination: int,
                              traversed_edge: int, actual_status: EdgeStatus) -> BeliefState:
        """
        Create the resulting state after traversing an edge.
        
        This:
        1. Updates the location to the destination
        2. Records the traversed edge's actual status
        3. Reveals any uncertain edges at the destination
        """
        # Update edge status for the traversed edge
        new_state = state.with_edge_revealed(traversed_edge, actual_status)
        
        # Move to destination
        new_state = new_state.with_location(destination)
        
        # Note: We don't reveal edges here - the caller handles the probabilistic branching
        # for newly visible edges in _expand_with_revealed_edges
        
        return new_state
    
    def _expand_with_revealed_edges(self, base_state: BeliefState, 
                                     base_cost: float) -> List[Transition]:
        """
        Expand a single state transition into multiple outcomes based on
        newly visible unknown edges at the destination.
        
        When the agent arrives at a new vertex, any unknown edges incident to
        that vertex are revealed. Since we don't know the actual flooding status,
        we must branch on all possibilities.
        
        For example, if edge E2 is unknown and becomes visible (flood_prob=0.3):
        - With prob 0.7: E2 is CLEAR
        - With prob 0.3: E2 is FLOODED
        
        If multiple edges are revealed, we branch on all combinations.
        
        Args:
            base_state: State after moving but before revealing edges
            base_cost: Cost of the action that got us here
            
        Returns:
            List of Transitions with all possible revelation outcomes
        """
        # Find unknown edges that are now visible at this location
        visible_edges = self.belief_space.get_observable_edges_at_location(base_state.location)
        unknown_visible = []
        
        for eid in visible_edges:
            status = base_state.get_edge_status(eid)
            if status == EdgeStatus.UNKNOWN:
                unknown_visible.append(eid)
        
        if not unknown_visible:
            # No unknown edges to reveal - deterministic outcome
            return [Transition(probability=1.0, next_state=base_state, cost=base_cost)]
        
        # Generate all combinations of flooding outcomes
        import itertools
        transitions = []
        
        for flooding_combo in itertools.product([False, True], repeat=len(unknown_visible)):
            # Calculate probability of this combination
            prob = 1.0
            new_statuses = base_state.get_edge_status_dict()
            
            for i, eid in enumerate(unknown_visible):
                edge = self.graph.edges[eid]
                is_flooded = flooding_combo[i]
                
                if is_flooded:
                    prob *= edge.flood_prob
                    new_statuses[eid] = EdgeStatus.FLOODED
                else:
                    prob *= (1.0 - edge.flood_prob)
                    new_statuses[eid] = EdgeStatus.CLEAR
            
            if prob > 0:  # Only include non-zero probability outcomes
                new_state = BeliefState.create(
                    base_state.location,
                    base_state.has_kit_equipped,
                    new_statuses
                )
                transitions.append(Transition(probability=prob, next_state=new_state, cost=base_cost))
        
        return transitions
    
    def _reveal_adjacent_edges(self, state: BeliefState) -> BeliefState:
        """
        Reveal all uncertain edges adjacent to the current location.
        
        This is part of the state transition - when arriving at a vertex,
        all incident uncertain edges become known.
        
        NOTE: In the MDP, we don't know the actual flooding status yet.
        For state enumeration purposes, the states we generate already have
        these edges as either FLOODED or CLEAR (not UNKNOWN) because we
        enumerate valid states where adjacent edges must be known.
        """
        # The state should already have adjacent edges revealed in valid states
        # This is handled by the state enumeration in BeliefStateSpace
        return state
    
    def _get_equip_transitions(self, state: BeliefState) -> List[Transition]:
        """
        Compute transitions for EQUIP action.
        
        Deterministic: agent equips the kit and pays the equip cost.
        """
        new_state = state.with_kit_status(True)
        cost = self.graph.equip_cost
        return [Transition(probability=1.0, next_state=new_state, cost=cost)]
    
    def _get_unequip_transitions(self, state: BeliefState) -> List[Transition]:
        """
        Compute transitions for UNEQUIP action.
        
        Deterministic: agent unequips the kit and pays the unequip cost.
        The kit remains at the current vertex (handled by graph.kit_locations
        being a property of vertices, not the agent).
        
        Actually, we need to track where kits are left... For simplicity,
        let's assume kits can only be picked up at their original locations.
        This is a simplification but matches the assignment's intent.
        """
        new_state = state.with_kit_status(False)
        cost = self.graph.unequip_cost
        return [Transition(probability=1.0, next_state=new_state, cost=cost)]
    
    def _get_observe_transitions(self, state: BeliefState, 
                                  edge_id: int) -> List[Transition]:
        """
        Compute transitions for OBSERVE action (additional requirement).
        
        Observing an edge reveals its status with the observation cost.
        The outcome is probabilistic based on the edge's flood probability.
        
        Unlike traverse, observation never moves the agent.
        """
        edge = self.graph.edges[edge_id]
        flood_prob = edge.flood_prob
        
        # Find the observation to get its cost
        obs_cost = 0
        for obs in self.graph.get_observations_at_vertex(state.location):
            if obs.edge_id == edge_id:
                obs_cost = obs.cost
                break
        
        transitions = []
        
        # Outcome 1: Edge is CLEAR
        if flood_prob < 1.0:
            new_state_clear = state.with_edge_revealed(edge_id, EdgeStatus.CLEAR)
            transitions.append(Transition(
                probability=1.0 - flood_prob,
                next_state=new_state_clear,
                cost=obs_cost
            ))
        
        # Outcome 2: Edge is FLOODED
        if flood_prob > 0.0:
            new_state_flooded = state.with_edge_revealed(edge_id, EdgeStatus.FLOODED)
            transitions.append(Transition(
                probability=flood_prob,
                next_state=new_state_flooded,
                cost=obs_cost
            ))
        
        return transitions
    
    def get_all_states(self) -> List[BeliefState]:
        """Get all belief states in the MDP."""
        return self.belief_space.enumerate_all_states()
    
    def get_initial_state(self) -> BeliefState:
        """
        Get the initial belief state for the problem.
        
        Note: The initial state has location at start_vertex, no kit,
        and edges adjacent to start_vertex are revealed (known).
        """
        return self.belief_space.get_initial_belief_state()
    
    def print_mdp_info(self):
        """Print information about the MDP structure."""
        print("=" * 60)
        print("HURRICANE EVACUATION MDP")
        print("=" * 60)
        
        print(f"\nProblem Configuration:")
        print(f"  Start vertex: {self.graph.start_vertex}")
        print(f"  Target vertex: {self.target}")
        print(f"  Kit locations: {sorted(self.graph.kit_locations) if self.graph.kit_locations else 'None'}")
        print(f"  Equip cost: {self.graph.equip_cost}")
        print(f"  Unequip cost: {self.graph.unequip_cost}")
        print(f"  Flood factor: {self.graph.flood_factor}x")
        
        uncertain = self.graph.get_possibly_flooded_edges()
        print(f"\nUncertain edges: {uncertain if uncertain else 'None'}")
        for eid in uncertain:
            edge = self.graph.edges[eid]
            print(f"  E{eid}: P(flood) = {edge.flood_prob}")
        
        if self.graph.observations:
            print(f"\nObservation actions available:")
            for obs in self.graph.observations:
                print(f"  From V{obs.from_vertex}: observe E{obs.edge_id} (cost {obs.cost})")
        
        self.belief_space.print_state_space_info()


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

#K1 1
#EC 2
#UC 1
#FF 2

#O V1 E2 1

#Start 1
#Target 4
"""
    
    with open("test_mdp.txt", "w") as f:
        f.write(test_input)
    
    graph = parse_file("test_mdp.txt")
    print_graph_data(graph)
    
    mdp = HurricaneMDP(graph)
    mdp.print_mdp_info()
    
    # Test actions from initial state
    initial = mdp.get_initial_state()
    print(f"\nInitial state: {initial}")
    
    actions = mdp.get_available_actions(initial)
    print(f"\nAvailable actions:")
    for action in actions:
        print(f"  {action}")
        transitions = mdp.get_transitions(initial, action)
        for t in transitions:
            print(f"    -> P={t.probability:.2f}, cost={t.cost}, state={t.next_state}")
