from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from belief_state import BeliefState, EdgeStatus
from mdp import HurricaneMDP, Action, ActionType, Transition


@dataclass
class PolicyEntry:
    optimal_action: Optional[Action]  # None for terminal states
    expected_value: float             # Expected cost to reach target
    is_reachable: bool               # Can this state be reached from start?


class ValueIterationSolver:
    def __init__(self, mdp: HurricaneMDP, epsilon: float = 1e-6, max_iterations: int = 10000):
        self.mdp = mdp
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        
        # Solution storage
        self.values: Dict[BeliefState, float] = {}
        self.policy: Dict[BeliefState, Optional[Action]] = {}
        self.reachable_states: Set[BeliefState] = set()
        
        # Statistics
        self.iterations_run = 0
        self.converged = False
    
    def solve(self, verbose: bool = True) -> Dict[BeliefState, PolicyEntry]:
        if verbose:
            print("\n" + "=" * 60)
            print("RUNNING VALUE ITERATION")
            print("=" * 60)
        
        # Get all states
        all_states = self.mdp.get_all_states()
        if verbose:
            print(f"Total belief states: {len(all_states)}")
        
        # Initialize values
        # Terminal states have value 0; others start at infinity
        for state in all_states:
            if self.mdp.is_terminal(state):
                self.values[state] = 0.0
                self.policy[state] = None
            else:
                self.values[state] = float('inf')  # Will be updated
                self.policy[state] = None
        
        # Value iteration main loop
        # We track convergence only for states that have finite values
        for iteration in range(self.max_iterations):
            max_delta = 0.0
            num_updated = 0
            num_finite = 0
            
            # Update each non-terminal state
            for state in all_states:
                if self.mdp.is_terminal(state):
                    continue
                
                old_value = self.values[state]
                new_value, best_action = self._compute_best_action(state)
                
                self.values[state] = new_value
                self.policy[state] = best_action
                
                # Track maximum change for convergence check
                # Only consider states that have finite values (reachable from goal)
                if old_value != float('inf') and new_value != float('inf'):
                    delta = abs(new_value - old_value)
                    max_delta = max(max_delta, delta)
                    num_finite += 1
                elif old_value == float('inf') and new_value != float('inf'):
                    # State became reachable - count this as an update
                    num_updated += 1
                    num_finite += 1
            
            self.iterations_run = iteration + 1
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: max_delta = {max_delta:.8f}, "
                      f"finite states = {num_finite}")
            
            # Check convergence: 
            # - max_delta is small for all finite states
            # - no new states became finite this iteration
            if max_delta < self.epsilon and num_updated == 0:
                self.converged = True
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break
        
        if not self.converged and verbose:
            print(f"  Warning: Did not converge after {self.max_iterations} iterations")
        
        # Compute reachable states
        self._compute_reachable_states()
        if verbose:
            print(f"Reachable states from start: {len(self.reachable_states)}")
        
        # Build result
        result = {}
        for state in all_states:
            result[state] = PolicyEntry(
                optimal_action=self.policy[state],
                expected_value=self.values[state],
                is_reachable=(state in self.reachable_states)
            )
        
        return result
    
    def _compute_best_action(self, state: BeliefState) -> Tuple[float, Optional[Action]]:
        actions = self.mdp.get_available_actions(state)
        
        if not actions:
            # No actions available - this is a dead end (shouldn't happen if graph is valid)
            return float('inf'), None
        
        best_value = float('inf')
        best_action = None
        
        for action in actions:
            q_value = self._compute_q_value(state, action)
            
            if q_value < best_value:
                best_value = q_value
                best_action = action
        
        return best_value, best_action
    
    def _compute_q_value(self, state: BeliefState, action: Action) -> float:
        transitions = self.mdp.get_transitions(state, action)
        
        expected_cost = 0.0
        for trans in transitions:
            # Cost of this transition plus value of next state
            future_cost = self.values.get(trans.next_state, float('inf'))
            expected_cost += trans.probability * (trans.cost + future_cost)
        
        return expected_cost
    
    def _compute_reachable_states(self):
        self.reachable_states = set()
        
        # We need to consider multiple possible initial states because
        # the edges visible from start can be flooded or clear
        initial_states = self._get_possible_initial_states()
        
        # BFS from all possible initial states
        queue = list(initial_states)
        visited = set(initial_states)
        
        while queue:
            state = queue.pop(0)
            self.reachable_states.add(state)
            
            # Get all possible next states from any action
            for action in self.mdp.get_available_actions(state):
                for trans in self.mdp.get_transitions(state, action):
                    if trans.next_state not in visited:
                        visited.add(trans.next_state)
                        queue.append(trans.next_state)
    
    def _get_possible_initial_states(self) -> List[BeliefState]:
        base_initial = self.mdp.get_initial_state()
        
        # Find edges that are revealed at the start location
        visible_edges = self.mdp.belief_space.get_observable_edges_at_location(
            self.mdp.graph.start_vertex
        )
        
        if not visible_edges:
            # No uncertain edges at start - only one initial state
            return [base_initial]
        
        # Generate all combinations of flooding for visible edges
        import itertools
        
        possible_states = []
        visible_list = sorted(visible_edges)
        
        for flooding_combo in itertools.product([False, True], repeat=len(visible_list)):
            # Build the state with this flooding configuration
            statuses = base_initial.get_edge_status_dict()
            
            for i, eid in enumerate(visible_list):
                if flooding_combo[i]:
                    statuses[eid] = EdgeStatus.FLOODED
                else:
                    statuses[eid] = EdgeStatus.CLEAR
            
            state = BeliefState.create(
                base_initial.location,
                base_initial.has_kit_equipped,
                statuses
            )
            possible_states.append(state)
        
        return possible_states
    
    def get_expected_cost_from_start(self) -> float:
        initial_states = self._get_possible_initial_states()
        
        if len(initial_states) == 1:
            return self.values.get(initial_states[0], float('inf'))
        
        # Compute probability and value for each initial state
        visible_edges = sorted(self.mdp.belief_space.get_observable_edges_at_location(
            self.mdp.graph.start_vertex
        ))
        
        total = 0.0
        for state in initial_states:
            # Compute probability of this initial state
            prob = 1.0
            for eid in visible_edges:
                edge = self.mdp.graph.edges[eid]
                status = state.get_edge_status(eid)
                if status == EdgeStatus.FLOODED:
                    prob *= edge.flood_prob
                else:
                    prob *= (1.0 - edge.flood_prob)
            
            total += prob * self.values.get(state, float('inf'))
        
        return total
    
    def print_policy(self, full: bool = False):
        print("\n" + "=" * 60)
        print("COMPUTED POLICY")
        print("=" * 60)
        
        print(f"\nExpected cost from start: {self.get_expected_cost_from_start():.4f}")
        print(f"Iterations: {self.iterations_run}")
        print(f"Converged: {self.converged}")
        
        states_to_print = self.mdp.get_all_states() if full else list(self.reachable_states)
        states_to_print = sorted(states_to_print, 
                                  key=lambda s: (s.location, s.has_kit_equipped, str(s.edge_status)))
        
        print(f"\nPolicy ({'all' if full else 'reachable'} states):")
        print("-" * 60)
        
        for state in states_to_print:
            value = self.values.get(state, float('inf'))
            action = self.policy.get(state)
            reachable = "  " if state in self.reachable_states else "X "
            
            if self.mdp.is_terminal(state):
                print(f"{reachable}{state}")
                print(f"        -> GOAL (cost: 0)")
            elif action is not None:
                print(f"{reachable}{state}")
                print(f"        -> {action} (expected cost: {value:.4f})")
            else:
                print(f"{reachable}{state}")
                print(f"        -> NO ACTION (expected cost: {value:.4f})")
        
        if not full:
            unreachable_count = len(self.mdp.get_all_states()) - len(self.reachable_states)
            print(f"\n({unreachable_count} unreachable states not shown)")


def print_state_values(solver: ValueIterationSolver, 
                       group_by_location: bool = True):
    print("\n" + "=" * 60)
    print("STATE VALUES")
    print("=" * 60)
    
    if group_by_location:
        # Group states by location
        by_location: Dict[int, List[BeliefState]] = {}
        for state in solver.mdp.get_all_states():
            loc = state.location
            if loc not in by_location:
                by_location[loc] = []
            by_location[loc].append(state)
        
        for loc in sorted(by_location.keys()):
            print(f"\nVertex {loc}:")
            states = sorted(by_location[loc], 
                           key=lambda s: (s.has_kit_equipped, str(s.edge_status)))
            
            for state in states:
                value = solver.values.get(state, float('inf'))
                action = solver.policy.get(state)
                reachable = "*" if state in solver.reachable_states else " "
                
                kit_str = "kit" if state.has_kit_equipped else "   "
                edge_str = ', '.join(f"E{eid}:{s.value[0]}" 
                                     for eid, s in state.edge_status)
                
                if solver.mdp.is_terminal(state):
                    print(f"  {reachable} [{kit_str}] [{edge_str}] -> GOAL")
                elif value == float('inf'):
                    print(f"  {reachable} [{kit_str}] [{edge_str}] -> UNREACHABLE")
                else:
                    action_str = str(action) if action else "NO ACTION"
                    print(f"  {reachable} [{kit_str}] [{edge_str}] -> {action_str} (V={value:.2f})")
    else:
        # Simple list
        for state in sorted(solver.mdp.get_all_states(), 
                           key=lambda s: (s.location, s.has_kit_equipped)):
            value = solver.values.get(state, float('inf'))
            print(f"  {state}: V = {value:.4f}")
