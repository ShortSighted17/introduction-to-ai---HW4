import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from belief_state import BeliefState, EdgeStatus, BeliefStateSpace
from mdp import HurricaneMDP, Action, ActionType
from value_iteration import ValueIterationSolver, PolicyEntry
from parser import GraphData


@dataclass
class WorldState:
    flooding: Dict[int, bool]
    kit_locations: Set[int]
    
    def is_flooded(self, edge_id: int) -> bool:
        return self.flooding.get(edge_id, False)
    
    def __repr__(self):
        flooded_edges = [eid for eid, flooded in self.flooding.items() if flooded]
        return f"World(flooded={flooded_edges}, kits={self.kit_locations})"


@dataclass 
class SimulationStep:
    state_before: BeliefState
    action: Action
    cost: float
    state_after: BeliefState
    observations: Dict[int, EdgeStatus]  # What edges were revealed and their status


class Simulator:
    def __init__(self, mdp: HurricaneMDP, solver: ValueIterationSolver):
        self.mdp = mdp
        self.solver = solver
        self.graph = mdp.graph
    
    def generate_world(self, seed: Optional[int] = None) -> WorldState:
        if seed is not None:
            random.seed(seed)
        
        flooding = {}
        for eid, edge in self.graph.edges.items():
            if edge.flood_prob > 0:
                flooding[eid] = random.random() < edge.flood_prob
            else:
                flooding[eid] = False
        
        # Copy initial kit locations
        kit_locations = set(self.graph.kit_locations)
        
        return WorldState(flooding, kit_locations)
    
    def get_initial_belief_state(self, world: WorldState) -> BeliefState:
        start = self.graph.start_vertex
        
        # Start with all uncertain edges as UNKNOWN
        uncertain_edges = self.graph.get_possibly_flooded_edges()
        statuses = {eid: EdgeStatus.UNKNOWN for eid in uncertain_edges}
        
        # Reveal edges visible from starting position
        visible = self.mdp.belief_space.get_observable_edges_at_location(start)
        for eid in visible:
            if eid in statuses:
                statuses[eid] = EdgeStatus.FLOODED if world.is_flooded(eid) else EdgeStatus.CLEAR
        
        return BeliefState.create(start, False, statuses)
    
    def run_simulation(self, world: Optional[WorldState] = None,
                       seed: Optional[int] = None,
                       max_steps: int = 1000,
                       verbose: bool = True) -> Tuple[List[SimulationStep], float, bool]:
        # Generate or use provided world
        if world is None:
            world = self.generate_world(seed)
        
        if verbose:
            print("\n" + "=" * 60)
            print("SIMULATION")
            print("=" * 60)
            print(f"\nWorld configuration: {world}")
            flooded = [f"E{eid}" for eid, f in world.flooding.items() if f]
            print(f"Flooded edges: {flooded if flooded else 'None'}")
        
        # Initialize
        state = self.get_initial_belief_state(world)
        steps = []
        total_cost = 0.0
        
        if verbose:
            print(f"\nStarting at: {state}")
        
        # Main simulation loop
        for step_num in range(max_steps):
            # Check if we've reached the goal
            if self.mdp.is_terminal(state):
                if verbose:
                    print(f"\n*** GOAL REACHED at vertex {state.location}! ***")
                    print(f"Total cost: {total_cost}")
                return steps, total_cost, True
            
            # Get the optimal action from policy
            action = self.solver.policy.get(state)
            
            if action is None:
                if verbose:
                    print(f"\n*** STUCK - No action available at {state} ***")
                return steps, total_cost, False
            
            # Execute the action
            new_state, cost, observations = self._execute_action(state, action, world)
            
            # Record the step
            step = SimulationStep(
                state_before=state,
                action=action,
                cost=cost,
                state_after=new_state,
                observations=observations
            )
            steps.append(step)
            total_cost += cost
            
            if verbose:
                self._print_step(step_num + 1, step)
            
            # Update state
            state = new_state
        
        if verbose:
            print(f"\n*** MAX STEPS REACHED ({max_steps}) ***")
        return steps, total_cost, False
    
    def _execute_action(self, state: BeliefState, action: Action, 
                        world: WorldState) -> Tuple[BeliefState, float, Dict[int, EdgeStatus]]:
        observations = {}
        
        if action.action_type == ActionType.TRAVERSE:
            return self._execute_traverse(state, action.edge_id, world)
        
        elif action.action_type == ActionType.EQUIP:
            new_state = state.with_kit_status(True)
            return new_state, self.graph.equip_cost, {}
        
        elif action.action_type == ActionType.UNEQUIP:
            new_state = state.with_kit_status(False)
            return new_state, self.graph.unequip_cost, {}
        
        elif action.action_type == ActionType.OBSERVE:
            return self._execute_observe(state, action.edge_id, world)
        
        return state, 0, {}
    
    def _execute_traverse(self, state: BeliefState, edge_id: int,
                          world: WorldState) -> Tuple[BeliefState, float, Dict[int, EdgeStatus]]:
        edge = self.graph.edges[edge_id]
        current_status = state.get_edge_status(edge_id)
        observations = {}
        
        # Determine actual status
        is_actually_flooded = world.is_flooded(edge_id)
        actual_status = EdgeStatus.FLOODED if is_actually_flooded else EdgeStatus.CLEAR
        
        # If edge was unknown, we now learn its status
        if current_status == EdgeStatus.UNKNOWN:
            observations[edge_id] = actual_status
            state = state.with_edge_revealed(edge_id, actual_status)
        
        # Check if we can traverse
        if is_actually_flooded and not state.has_kit_equipped:
            # BLOCKED - can't traverse without kit
            # We learned the edge is flooded, but stay in place
            return state, 0, observations
        
        # Calculate cost
        cost = self.graph.get_traversal_cost(edge, state.has_kit_equipped)
        
        # Move to destination
        destination = edge.get_other_endpoint(state.location)
        new_state = state.with_location(destination)
        
        # Reveal edges at new location
        visible_at_dest = self.mdp.belief_space.get_observable_edges_at_location(destination)
        for eid in visible_at_dest:
            current = new_state.get_edge_status(eid)
            if current == EdgeStatus.UNKNOWN:
                is_flooded = world.is_flooded(eid)
                revealed_status = EdgeStatus.FLOODED if is_flooded else EdgeStatus.CLEAR
                new_state = new_state.with_edge_revealed(eid, revealed_status)
                observations[eid] = revealed_status
        
        return new_state, cost, observations
    
    def _execute_observe(self, state: BeliefState, edge_id: int,
                         world: WorldState) -> Tuple[BeliefState, float, Dict[int, EdgeStatus]]:
        # Find the observation cost
        obs_cost = 0
        for obs in self.graph.get_observations_at_vertex(state.location):
            if obs.edge_id == edge_id:
                obs_cost = obs.cost
                break
        
        # Reveal the edge status
        is_flooded = world.is_flooded(edge_id)
        actual_status = EdgeStatus.FLOODED if is_flooded else EdgeStatus.CLEAR
        
        new_state = state.with_edge_revealed(edge_id, actual_status)
        observations = {edge_id: actual_status}
        
        return new_state, obs_cost, observations
    
    def _print_step(self, step_num: int, step: SimulationStep):
        print(f"\nStep {step_num}:")
        print(f"  State: V{step.state_before.location}, "
              f"kit={'yes' if step.state_before.has_kit_equipped else 'no'}")
        print(f"  Action: {step.action}")
        print(f"  Cost: {step.cost}")
        
        if step.observations:
            obs_str = ', '.join(f"E{eid}={s.value}" for eid, s in step.observations.items())
            print(f"  Observations: {obs_str}")
        
        print(f"  New state: V{step.state_after.location}, "
              f"kit={'yes' if step.state_after.has_kit_equipped else 'no'}")
    
    def run_multiple_simulations(self, num_simulations: int = 100,
                                  verbose: bool = False) -> Dict:
        costs = []
        successes = 0
        failures = 0
        
        for i in range(num_simulations):
            _, cost, success = self.run_simulation(verbose=verbose)
            
            if success:
                costs.append(cost)
                successes += 1
            else:
                failures += 1
        
        avg_cost = sum(costs) / len(costs) if costs else float('inf')
        
        return {
            'num_simulations': num_simulations,
            'successes': successes,
            'failures': failures,
            'success_rate': successes / num_simulations if num_simulations > 0 else 0,
            'average_cost': avg_cost,
            'min_cost': min(costs) if costs else float('inf'),
            'max_cost': max(costs) if costs else float('inf'),
            'all_costs': costs
        }


def print_world_state(world: WorldState, graph: GraphData):
    print("\n" + "-" * 40)
    print("WORLD STATE (Ground Truth)")
    print("-" * 40)
    
    print("\nEdge flooding status:")
    for eid, edge in sorted(graph.edges.items()):
        if edge.flood_prob > 0:
            status = "FLOODED" if world.flooding.get(eid, False) else "clear"
            prob_str = f"(prob was {edge.flood_prob})"
            print(f"  E{eid} ({edge.u}-{edge.v}): {status} {prob_str}")
    
    print(f"\nKit locations: {sorted(world.kit_locations) if world.kit_locations else 'None'}")
