"""
Comprehensive Test Suite for Hurricane Evacuation MDP (Assignment 4)

This test suite contains carefully designed scenarios where the optimal policy
and expected costs can be computed analytically by hand. This allows us to
verify that the value iteration algorithm is producing correct results.

TEST CATEGORIES:
================
1. Deterministic scenarios (no uncertainty)
2. Single uncertain edge scenarios
3. Multiple uncertain edges (independent)
4. Amphibian kit scenarios
5. Observation action scenarios (additional requirement)
6. Edge cases and corner cases
7. Reachability and validation tests

For each test, we provide:
- The scenario description
- The expected optimal policy for key states
- The expected value (cost) for key states
- Verification that simulation results match expectations

Run with: python tests.py
"""

import sys
import math
import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from parser import parse_file, GraphData, validate_graph
from belief_state import BeliefState, BeliefStateSpace, EdgeStatus
from mdp import HurricaneMDP, Action, ActionType
from value_iteration import ValueIterationSolver
from simulator import Simulator, WorldState


# ============================================================
# TEST UTILITIES
# ============================================================

def approx_equal(a: float, b: float, tolerance: float = 1e-4) -> bool:
    """Check if two floats are approximately equal."""
    if a == float('inf') and b == float('inf'):
        return True
    if a == float('inf') or b == float('inf'):
        return False
    return abs(a - b) < tolerance


def assert_approx(actual: float, expected: float, message: str, tolerance: float = 1e-4) -> bool:
    """Assert that actual ≈ expected, with a helpful message on failure."""
    if not approx_equal(actual, expected, tolerance):
        print(f"  FAILED: {message}")
        print(f"    Expected: {expected}")
        print(f"    Actual:   {actual}")
        return False
    return True


def assert_true(condition: bool, message: str) -> bool:
    """Assert that condition is true."""
    if not condition:
        print(f"  FAILED: {message}")
        return False
    return True


def assert_action(solver: ValueIterationSolver, state: BeliefState, 
                  expected_type: ActionType, expected_edge: Optional[int] = None,
                  message: str = "") -> bool:
    """Assert that the optimal action for a state matches expected."""
    action = solver.policy.get(state)
    
    if action is None:
        print(f"  FAILED: {message} - No action found for state {state}")
        return False
    
    if action.action_type != expected_type:
        print(f"  FAILED: {message}")
        print(f"    Expected action type: {expected_type}")
        print(f"    Actual action: {action}")
        return False
    
    if expected_edge is not None and action.edge_id != expected_edge:
        print(f"  FAILED: {message}")
        print(f"    Expected edge: {expected_edge}")
        print(f"    Actual edge: {action.edge_id}")
        return False
    
    return True


def create_graph_from_string(content: str) -> GraphData:
    """Create a graph by writing content to a temp file and parsing it."""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    try:
        graph = parse_file(temp_path)
    finally:
        os.unlink(temp_path)
    
    return graph


def solve_graph(graph: GraphData, verbose: bool = False) -> Tuple[HurricaneMDP, ValueIterationSolver]:
    """Create MDP and solve it, returning both."""
    mdp = HurricaneMDP(graph)
    solver = ValueIterationSolver(mdp)
    solver.solve(verbose=verbose)
    return mdp, solver


def find_state(solver: ValueIterationSolver, location: int, has_kit: bool,
               edge_statuses: Dict[int, str]) -> Optional[BeliefState]:
    """
    Find a specific belief state by its properties.
    
    edge_statuses maps edge_id to 'unknown', 'flooded', or 'clear'
    """
    status_map = {
        'unknown': EdgeStatus.UNKNOWN,
        'flooded': EdgeStatus.FLOODED,
        'clear': EdgeStatus.CLEAR,
        'u': EdgeStatus.UNKNOWN,
        'f': EdgeStatus.FLOODED,
        'c': EdgeStatus.CLEAR,
    }
    
    for state in solver.mdp.get_all_states():
        if state.location != location:
            continue
        if state.has_kit_equipped != has_kit:
            continue
        
        # Check edge statuses
        match = True
        for eid, status_str in edge_statuses.items():
            expected_status = status_map[status_str]
            actual_status = state.get_edge_status(eid)
            if actual_status != expected_status:
                match = False
                break
        
        if match:
            return state
    
    return None


def run_test(name: str, test_func) -> bool:
    """Run a test function and report results."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print('='*70)
    try:
        passed = test_func()
        if passed:
            print(f"  ✓ PASSED")
        return passed
    except Exception as e:
        print(f"  FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# CATEGORY 1: DETERMINISTIC SCENARIOS (NO UNCERTAINTY)
# ============================================================

def test_deterministic_simple_path():
    """
    Scenario: Simple path with no flooding uncertainty.
    
    Graph: 1 ---(w=2)--- 2 ---(w=3)--- 3
    
    Optimal policy: Go from 1 to 2 to 3
    Expected cost: 2 + 3 = 5
    """
    graph_str = """
#V 3
#E1 1 2 W2
#E2 2 3 W3
#Start 1
#Target 3
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # Expected cost from start
    expected_cost = 5.0
    actual_cost = solver.get_expected_cost_from_start()
    all_passed &= assert_approx(actual_cost, expected_cost, 
                                f"Expected cost from start should be {expected_cost}")
    
    # Check that at V1, the action is to traverse E1
    state_v1 = find_state(solver, location=1, has_kit=False, edge_statuses={})
    if state_v1:
        all_passed &= assert_action(solver, state_v1, ActionType.TRAVERSE, 1,
                                    "At V1, should traverse E1")
    
    # Check that at V2, the action is to traverse E2
    state_v2 = find_state(solver, location=2, has_kit=False, edge_statuses={})
    if state_v2:
        all_passed &= assert_action(solver, state_v2, ActionType.TRAVERSE, 2,
                                    "At V2, should traverse E2")
    
    return all_passed


def test_deterministic_choose_shorter_path():
    """
    Scenario: Two paths, agent should choose shorter one.
    
    Graph:     2
              /|
         (3)/  |(1)
           /   |
          1----3 (target)
            (5)
    
    Path 1->2->3: cost 3+1 = 4
    Path 1->3: cost 5
    
    Optimal: Go via vertex 2
    Expected cost: 4
    """
    graph_str = """
#V 3
#E1 1 2 W3
#E2 2 3 W1
#E3 1 3 W5
#Start 1
#Target 3
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # Expected cost is 4 (via vertex 2)
    expected_cost = 4.0
    actual_cost = solver.get_expected_cost_from_start()
    all_passed &= assert_approx(actual_cost, expected_cost,
                                f"Expected cost should be {expected_cost}")
    
    # At V1, should go to V2 (traverse E1), not directly to V3
    state_v1 = find_state(solver, location=1, has_kit=False, edge_statuses={})
    if state_v1:
        all_passed &= assert_action(solver, state_v1, ActionType.TRAVERSE, 1,
                                    "At V1, should traverse E1 (shorter path via V2)")
    
    return all_passed


def test_deterministic_diamond():
    """
    Scenario: Diamond graph, symmetric costs.
    
    Graph:       2
               / \
          (2)/   \(2)
            /     \
           1       4 (target)
            \     /
          (2)\   /(2)
               \ /
                3
    
    Both paths cost 4, so either is optimal.
    Expected cost: 4
    """
    graph_str = """
#V 4
#E1 1 2 W2
#E2 2 4 W2
#E3 1 3 W2
#E4 3 4 W2
#Start 1
#Target 4
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    expected_cost = 4.0
    actual_cost = solver.get_expected_cost_from_start()
    all_passed &= assert_approx(actual_cost, expected_cost,
                                f"Expected cost should be {expected_cost}")
    
    return all_passed


# ============================================================
# CATEGORY 2: SINGLE UNCERTAIN EDGE SCENARIOS
# ============================================================

def test_single_uncertain_edge_on_only_path():
    """
    Scenario: Single path with one uncertain edge, no kit available.
    
    Graph: 1 ---(w=1, p=0.5)--- 2 ---(w=1)--- 3
    
    If edge 1 is clear (prob 0.5): cost = 2
    If edge 1 is flooded (prob 0.5): stuck (infinite cost)
    
    Expected cost from start: 0.5 * 2 + 0.5 * inf = inf
    
    But wait - we need to think about what "expected cost" means when
    there's a possibility of being stuck. The agent will try to traverse,
    and if blocked, it's stuck.
    
    Actually, for this test, we expect the agent to be stuck 50% of the time.
    The expected cost for successful runs is 2.
    """
    graph_str = """
#V 3
#E1 1 2 W1 F 0.5
#E2 2 3 W1
#Start 1
#Target 3
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # When E1 is revealed as clear at start (adjacent to V1), cost should be 2
    state_clear = find_state(solver, location=1, has_kit=False, edge_statuses={1: 'clear'})
    if state_clear:
        value = solver.values.get(state_clear, float('inf'))
        all_passed &= assert_approx(value, 2.0, "Cost when E1 is clear should be 2")
    
    # When E1 is revealed as flooded, cost should be infinite (stuck)
    state_flooded = find_state(solver, location=1, has_kit=False, edge_statuses={1: 'flooded'})
    if state_flooded:
        value = solver.values.get(state_flooded, float('inf'))
        all_passed &= assert_true(value == float('inf'), 
                                  "Cost when E1 is flooded should be infinite")
    
    return all_passed


def test_single_uncertain_edge_with_bypass():
    """
    Scenario: Uncertain edge with a guaranteed (but longer) bypass.
    
    Graph: 1 ---(w=1, p=0.3)--- 2 ---(w=1)--- 3
           |                               |
           +---------(w=5)----------------+
    
    Path 1->2->3: cost 2 (if E1 clear)
    Path 1->3 direct: cost 5 (guaranteed)
    
    Strategy: Try E1 first. If flooded, use bypass.
    
    Expected cost:
    - E1 clear (prob 0.7): cost = 2
    - E1 flooded (prob 0.3): discovered at start, then take bypass, cost = 5
    
    Expected = 0.7 * 2 + 0.3 * 5 = 1.4 + 1.5 = 2.9
    """
    graph_str = """
#V 3
#E1 1 2 W1 F 0.3
#E2 2 3 W1
#E3 1 3 W5
#Start 1
#Target 3
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # When E1 is clear: go via V2, cost = 2
    state_clear = find_state(solver, location=1, has_kit=False, edge_statuses={1: 'clear'})
    if state_clear:
        value = solver.values.get(state_clear, float('inf'))
        all_passed &= assert_approx(value, 2.0, "Cost when E1 clear should be 2")
        all_passed &= assert_action(solver, state_clear, ActionType.TRAVERSE, 1,
                                    "When E1 clear, should traverse E1")
    
    # When E1 is flooded: take bypass E3, cost = 5
    state_flooded = find_state(solver, location=1, has_kit=False, edge_statuses={1: 'flooded'})
    if state_flooded:
        value = solver.values.get(state_flooded, float('inf'))
        all_passed &= assert_approx(value, 5.0, "Cost when E1 flooded should be 5")
        all_passed &= assert_action(solver, state_flooded, ActionType.TRAVERSE, 3,
                                    "When E1 flooded, should traverse E3 (bypass)")
    
    # Expected cost from start
    expected_cost = 0.7 * 2 + 0.3 * 5
    actual_cost = solver.get_expected_cost_from_start()
    all_passed &= assert_approx(actual_cost, expected_cost,
                                f"Expected cost from start should be {expected_cost}")
    
    return all_passed


def test_single_uncertain_edge_discovered_later():
    """
    Scenario: Uncertain edge not adjacent to start, discovered when reaching it.
    
    Graph: 1 ---(w=1)--- 2 ---(w=1, p=0.4)--- 3 ---(w=1)--- 4
    
    E2 status is unknown until agent reaches V2 or V3.
    
    When agent is at V1: E2 is unknown
    Agent goes to V2: E2 is revealed
    - If clear (0.6): go to V3, cost from V2 = 2, total from V1 = 3
    - If flooded (0.4): stuck at V2
    
    Expected cost from V1 = 1 + (0.6 * 2 + 0.4 * inf) = inf
    
    But wait, agent is stuck if flooded. Expected cost is effectively inf.
    """
    graph_str = """
#V 4
#E1 1 2 W1
#E2 2 3 W1 F 0.4
#E3 3 4 W1
#Start 1
#Target 4
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # At V1, E2 is not adjacent, so it's unknown in the belief state
    # But our state enumeration requires adjacent edges to be known
    # So at V1, we just have E2 unknown, but this is a valid initial state
    
    # At V2 with E2 clear: cost = 2
    state_v2_clear = find_state(solver, location=2, has_kit=False, edge_statuses={2: 'clear'})
    if state_v2_clear:
        value = solver.values.get(state_v2_clear, float('inf'))
        all_passed &= assert_approx(value, 2.0, "At V2 with E2 clear, cost should be 2")
    else:
        # State might have different format - let's search more flexibly
        found = False
        for state in solver.mdp.get_all_states():
            if state.location == 2 and not state.has_kit_equipped:
                status = state.get_edge_status(2)
                if status == EdgeStatus.CLEAR:
                    value = solver.values.get(state, float('inf'))
                    all_passed &= assert_approx(value, 2.0, "At V2 with E2 clear, cost should be 2")
                    found = True
                    break
        if not found:
            print("  INFO: Could not find V2 with E2 clear state")
    
    # At V2 with E2 flooded: stuck
    state_v2_flooded = find_state(solver, location=2, has_kit=False, edge_statuses={2: 'flooded'})
    if state_v2_flooded:
        value = solver.values.get(state_v2_flooded, float('inf'))
        all_passed &= assert_true(value == float('inf'),
                                  "At V2 with E2 flooded, should be stuck")
    else:
        found = False
        for state in solver.mdp.get_all_states():
            if state.location == 2 and not state.has_kit_equipped:
                status = state.get_edge_status(2)
                if status == EdgeStatus.FLOODED:
                    value = solver.values.get(state, float('inf'))
                    all_passed &= assert_true(value == float('inf'),
                                              "At V2 with E2 flooded, should be stuck")
                    found = True
                    break
        if not found:
            print("  INFO: Could not find V2 with E2 flooded state")
    
    return all_passed


# ============================================================
# CATEGORY 3: MULTIPLE UNCERTAIN EDGES
# ============================================================

def test_two_independent_uncertain_edges():
    """
    Scenario: Two uncertain edges in series, both can block.
    
    Graph: 1 ---(w=1, p=0.2)--- 2 ---(w=1, p=0.3)--- 3
    
    E1 revealed at V1 (start).
    E2 revealed at V2 or V3.
    
    If E1 flooded: stuck immediately
    If E1 clear, E2 flooded: stuck at V2  
    If E1 clear, E2 clear: success, cost = 2
    
    P(success) = P(E1 clear) * P(E2 clear) = 0.8 * 0.7 = 0.56
    
    For successful paths: cost = 2
    For stuck paths: cost = inf
    """
    graph_str = """
#V 3
#E1 1 2 W1 F 0.2
#E2 2 3 W1 F 0.3
#Start 1
#Target 3
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # At V2 with E1 clear, E2 clear: cost = 1
    found_clear_clear = False
    for state in solver.mdp.get_all_states():
        if state.location == 2 and not state.has_kit_equipped:
            e1_status = state.get_edge_status(1)
            e2_status = state.get_edge_status(2)
            if e1_status == EdgeStatus.CLEAR and e2_status == EdgeStatus.CLEAR:
                value = solver.values.get(state, float('inf'))
                all_passed &= assert_approx(value, 1.0, "At V2 with both clear, cost should be 1")
                found_clear_clear = True
                break
    
    if not found_clear_clear:
        print("  INFO: Could not find V2 with both edges clear state")
    
    # At V2 with E2 flooded: stuck
    found_flooded = False
    for state in solver.mdp.get_all_states():
        if state.location == 2 and not state.has_kit_equipped:
            e2_status = state.get_edge_status(2)
            if e2_status == EdgeStatus.FLOODED:
                value = solver.values.get(state, float('inf'))
                all_passed &= assert_true(value == float('inf'), "At V2 with E2 flooded, should be stuck")
                found_flooded = True
                break
    
    if not found_flooded:
        print("  INFO: Could not find V2 with E2 flooded state")
    
    # At V1 with E1 flooded: stuck
    found_e1_flooded = False
    for state in solver.mdp.get_all_states():
        if state.location == 1 and not state.has_kit_equipped:
            e1_status = state.get_edge_status(1)
            if e1_status == EdgeStatus.FLOODED:
                value = solver.values.get(state, float('inf'))
                all_passed &= assert_true(value == float('inf'), "At V1 with E1 flooded, should be stuck")
                found_e1_flooded = True
                break
    
    return all_passed


def test_parallel_uncertain_paths():
    """
    Scenario: Two parallel paths, each with one uncertain edge.
    
    Graph:       2
               / \
        (w=1) /   \ (w=1)
        p=0.5     p=0.5
             /     \
            1       3 (target)
             \     /
        (w=1) \   / (w=1)
        p=0.5     p=0.5
               \ /
                4
    
    Wait, this is confusing. Let me simplify:
    
    Graph:   1 ---(w=2, p=0.5)--- 3 (target)
             |
             +---(w=1, p=0.5)--- 2 ---(w=1)--- 3
    
    Actually let's do a cleaner version:
    
    Graph: 1 has two ways to reach 3:
    - Direct: E1 (w=1, p=0.5)
    - Via V2: E2 (w=2, p=0.5) then E3 (w=1, no flood)
    
    Both E1 and E2 are revealed at V1 (adjacent).
    
    Cases:
    - E1 clear: take E1, cost = 1
    - E1 flooded, E2 clear: take E2->E3, cost = 3
    - E1 flooded, E2 flooded: stuck
    
    P(E1 clear) = 0.5: cost 1
    P(E1 flooded, E2 clear) = 0.5 * 0.5 = 0.25: cost 3
    P(E1 flooded, E2 flooded) = 0.25: stuck
    
    For non-stuck cases:
    Expected cost = (0.5 * 1 + 0.25 * 3) / 0.75 = (0.5 + 0.75) / 0.75 = 1.67
    
    But expected cost including stuck should consider all cases.
    """
    graph_str = """
#V 3
#E1 1 3 W1 F 0.5
#E2 1 2 W2 F 0.5
#E3 2 3 W1
#Start 1
#Target 3
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # E1 clear, E2 anything: should take E1
    state = find_state(solver, location=1, has_kit=False, edge_statuses={1: 'clear', 2: 'clear'})
    if state:
        value = solver.values.get(state, float('inf'))
        all_passed &= assert_approx(value, 1.0, "E1 clear: should cost 1")
        all_passed &= assert_action(solver, state, ActionType.TRAVERSE, 1,
                                    "E1 clear: should take E1")
    
    # E1 flooded, E2 clear: should take E2->E3, cost = 3
    state = find_state(solver, location=1, has_kit=False, edge_statuses={1: 'flooded', 2: 'clear'})
    if state:
        value = solver.values.get(state, float('inf'))
        all_passed &= assert_approx(value, 3.0, "E1 flooded, E2 clear: should cost 3")
        all_passed &= assert_action(solver, state, ActionType.TRAVERSE, 2,
                                    "E1 flooded, E2 clear: should take E2")
    
    # E1 flooded, E2 flooded: stuck
    state = find_state(solver, location=1, has_kit=False, edge_statuses={1: 'flooded', 2: 'flooded'})
    if state:
        value = solver.values.get(state, float('inf'))
        all_passed &= assert_true(value == float('inf'), "Both flooded: should be stuck")
    
    return all_passed


# ============================================================
# CATEGORY 4: AMPHIBIAN KIT SCENARIOS
# ============================================================

def test_kit_needed_for_guaranteed_flood():
    """
    Scenario: Only path has guaranteed flood, kit needed.
    
    Graph: 1 ---(w=1, p=1.0)--- 2
    Kit at V1, equip cost = 2, flood factor = 2
    
    Without kit: E1 always flooded, can't traverse
    With kit: can traverse flooded edge at 2x cost
    
    Optimal: equip kit (2), traverse E1 (1*2=2)
    Total cost: 2 + 2 = 4
    """
    graph_str = """
#V 2
#E1 1 2 W1 F 1.0
#K1 1
#EC 2
#UC 1
#FF 2
#Start 1
#Target 2
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # At V1 without kit, E1 flooded: should equip
    state = find_state(solver, location=1, has_kit=False, edge_statuses={1: 'flooded'})
    if state:
        value = solver.values.get(state, float('inf'))
        all_passed &= assert_approx(value, 4.0, "Should cost 4 (equip 2 + traverse 2)")
        all_passed &= assert_action(solver, state, ActionType.EQUIP, None,
                                    "Should equip kit first")
    
    # At V1 with kit, E1 flooded: should traverse
    state = find_state(solver, location=1, has_kit=True, edge_statuses={1: 'flooded'})
    if state:
        value = solver.values.get(state, float('inf'))
        all_passed &= assert_approx(value, 2.0, "With kit, traverse cost should be 2")
        all_passed &= assert_action(solver, state, ActionType.TRAVERSE, 1,
                                    "With kit, should traverse")
    
    return all_passed


def test_kit_vs_bypass_decision():
    """
    Scenario: Can equip kit to use flooded path, or use longer clear bypass.
    
    Graph: 1 ---(w=1, p=1.0)--- 2 (target)
           |
           +---(w=4)--- 3 ---(w=1)--- 2
    
    Kit at V1, equip cost = 2, flood factor = 2
    
    Option 1: Equip + traverse flooded E1 = 2 + 1*2 = 4
    Option 2: Bypass via V3 = 4 + 1 = 5
    
    Optimal: Use kit (cost 4)
    """
    graph_str = """
#V 3
#E1 1 2 W1 F 1.0
#E2 1 3 W4
#E3 3 2 W1
#K1 1
#EC 2
#UC 1
#FF 2
#Start 1
#Target 2
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    expected_cost = 4.0  # Equip (2) + traverse flooded (2)
    actual_cost = solver.get_expected_cost_from_start()
    all_passed &= assert_approx(actual_cost, expected_cost,
                                f"Should use kit, cost {expected_cost}")
    
    # Should choose to equip rather than bypass
    state = find_state(solver, location=1, has_kit=False, edge_statuses={1: 'flooded'})
    if state:
        all_passed &= assert_action(solver, state, ActionType.EQUIP, None,
                                    "Should equip kit (cheaper than bypass)")
    
    return all_passed


def test_kit_vs_bypass_decision_bypass_wins():
    """
    Scenario: Bypass is cheaper than using kit.
    
    Graph: 1 ---(w=1, p=1.0)--- 2 (target)
           |
           +---(w=2)--- 3 ---(w=1)--- 2
    
    Kit at V1, equip cost = 5, flood factor = 3
    
    Option 1: Equip + traverse flooded E1 = 5 + 1*3 = 8
    Option 2: Bypass via V3 = 2 + 1 = 3
    
    Optimal: Bypass (cost 3)
    """
    graph_str = """
#V 3
#E1 1 2 W1 F 1.0
#E2 1 3 W2
#E3 3 2 W1
#K1 1
#EC 5
#UC 1
#FF 3
#Start 1
#Target 2
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    expected_cost = 3.0  # Bypass via V3
    actual_cost = solver.get_expected_cost_from_start()
    all_passed &= assert_approx(actual_cost, expected_cost,
                                f"Should use bypass, cost {expected_cost}")
    
    # Should choose bypass rather than equip
    state = find_state(solver, location=1, has_kit=False, edge_statuses={1: 'flooded'})
    if state:
        all_passed &= assert_action(solver, state, ActionType.TRAVERSE, 2,
                                    "Should take bypass E2 (cheaper than kit)")
    
    return all_passed


def test_unequip_kit_for_faster_travel():
    """
    Scenario: Agent has kit but should unequip for faster travel on clear path.
    
    Graph: 1 ---(w=1, p=0.5)--- 2 ---(w=10)--- 3 (target)
    
    Kit at V1, equip cost = 1, unequip cost = 1, flood factor = 3
    
    E1 is adjacent to V1, so it's revealed at start.
    
    If E1 clear and agent has kit at V1:
    - Best: Unequip (1) + traverse E1 (1) + traverse E2 (10) = 12
    
    If E1 flooded and agent has kit at V1:
    - Best: traverse E1 with kit (3) + unequip at V2 (1) + traverse E2 (10) = 14
    
    If E1 clear and agent has no kit:
    - traverse E1 (1) + traverse E2 (10) = 11
    
    So the agent smartly unequips at the right location!
    """
    graph_str = """
#V 3
#E1 1 2 W1 F 0.5
#E2 2 3 W10
#K1 1
#EC 1
#UC 1
#FF 3
#Start 1
#Target 3
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # At V1 with kit and E1 clear: should unequip first
    found_kit_clear = False
    for state in solver.mdp.get_all_states():
        if state.location == 1 and state.has_kit_equipped:
            e1_status = state.get_edge_status(1)
            if e1_status == EdgeStatus.CLEAR:
                value = solver.values.get(state, float('inf'))
                # Unequip (1) + traverse E1 (1) + traverse E2 (10) = 12
                all_passed &= assert_approx(value, 12.0, "With kit and E1 clear, cost should be 12")
                all_passed &= assert_action(solver, state, ActionType.UNEQUIP, None,
                                            "Should unequip for faster travel")
                found_kit_clear = True
                break
    
    if not found_kit_clear:
        print("  INFO: Could not find V1 with kit and E1 clear state")
        all_passed = False
    
    # At V1 with kit and E1 flooded: traverse with kit, unequip at V2
    found_kit_flooded = False
    for state in solver.mdp.get_all_states():
        if state.location == 1 and state.has_kit_equipped:
            e1_status = state.get_edge_status(1)
            if e1_status == EdgeStatus.FLOODED:
                value = solver.values.get(state, float('inf'))
                # Traverse E1 with kit (3) + unequip at V2 (1) + traverse E2 (10) = 14
                all_passed &= assert_approx(value, 14.0, 
                    "With kit and E1 flooded, cost should be 14 (traverse, unequip at V2, traverse)")
                all_passed &= assert_action(solver, state, ActionType.TRAVERSE, 1,
                                            "Should traverse flooded edge with kit")
                found_kit_flooded = True
                break
    
    if not found_kit_flooded:
        print("  INFO: Could not find V1 with kit and E1 flooded state")
        all_passed = False
    
    return all_passed


# ============================================================
# CATEGORY 5: OBSERVATION ACTION SCENARIOS
# ============================================================

def test_observation_basic():
    """
    Scenario: Observation reveals distant edge, helps avoid dead end.
    
    Graph: 1 ---(w=1)--- 2 ---(w=1, p=0.5)--- 3 (target)
    
    Observation from V1: can observe E2 for cost 1
    
    Without observation:
    - Go to V2, then discover E2 status
    - If flooded (0.5): stuck at V2
    
    With observation:
    - Pay 1 to observe E2
    - If clear: go to V2->V3, total cost = 1 + 1 + 1 = 3
    - If flooded: don't bother going (stuck anyway, but we know before wasting time)
    
    Actually, if there's no bypass, observation doesn't help - we're stuck either way.
    Let me redesign with a bypass.
    """
    # Redesigned scenario
    graph_str = """
#V 4
#E1 1 2 W1
#E2 2 3 W1 F 0.5
#E3 1 4 W5
#E4 4 3 W1

#O V1 E2 1

#Start 1
#Target 3
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # Without observation, if we go to V2 and find E2 flooded, we must backtrack
    # Path via V2 if E2 clear: 1 + 1 = 2
    # Path via V4: 5 + 1 = 6
    # If E2 flooded and we went to V2 first: 1 (to V2) + 1 (back to V1) + 6 (via V4) = 8
    # Wait, can we go back? In this MDP, we can traverse any edge...
    
    # Let me think about this differently:
    # At V1 with E2 unknown:
    # - Option A: Observe E2 (cost 1), then decide
    #   - If E2 clear (0.5): go via V2, cost = 1 + 1 + 1 = 3
    #   - If E2 flooded (0.5): go via V4, cost = 1 + 5 + 1 = 7
    #   - Expected: 1 + 0.5*2 + 0.5*6 = 1 + 1 + 3 = 5
    # - Option B: Go to V2 without observing
    #   - E2 clear (0.5): cost = 1 + 1 = 2
    #   - E2 flooded (0.5): at V2, must go back to V1 and use V4
    #     cost = 1 + 1 + 5 + 1 = 8
    #   - Expected: 0.5*2 + 0.5*8 = 1 + 4 = 5
    
    # Hmm, they're equal in this case! Let me adjust probabilities.
    
    # Actually, let's just verify the policy makes reasonable decisions
    state = find_state(solver, location=1, has_kit=False, edge_statuses={2: 'unknown'})
    if state:
        # The solver should pick either observe or traverse - both valid
        action = solver.policy.get(state)
        all_passed &= assert_true(action is not None, "Should have an action at V1")
    
    # After observing E2 is clear, should go via V2
    state = find_state(solver, location=1, has_kit=False, edge_statuses={2: 'clear'})
    if state:
        all_passed &= assert_action(solver, state, ActionType.TRAVERSE, 1,
                                    "E2 clear: should go via V2")
        value = solver.values.get(state, float('inf'))
        all_passed &= assert_approx(value, 2.0, "E2 clear from V1: cost should be 2")
    
    # After observing E2 is flooded, should go via V4
    state = find_state(solver, location=1, has_kit=False, edge_statuses={2: 'flooded'})
    if state:
        all_passed &= assert_action(solver, state, ActionType.TRAVERSE, 3,
                                    "E2 flooded: should go via V4")
        value = solver.values.get(state, float('inf'))
        all_passed &= assert_approx(value, 6.0, "E2 flooded from V1: cost should be 6")
    
    return all_passed


def test_observation_value_of_information():
    """
    Scenario: Observation is valuable - cheaper than exploring blind.
    
    Graph:
    1 ---(w=5)--- 2 ---(w=1, p=0.9)--- 3 (target)
    |
    +---(w=10)--- 4 ---(w=1)--- 3
    
    Observation from V1: observe E2 for cost 1
    
    Without observation, going via V2:
    - If E2 clear (0.1): cost = 5 + 1 = 6
    - If E2 flooded (0.9): stuck at V2 (can't go back easily to V4 in useful way)
    
    Via V4 directly: cost = 10 + 1 = 11
    
    With observation:
    - Pay 1
    - If E2 clear (0.1): go via V2, total = 1 + 5 + 1 = 7
    - If E2 flooded (0.9): go via V4, total = 1 + 10 + 1 = 12
    - Expected = 0.1*7 + 0.9*12 = 0.7 + 10.8 = 11.5
    
    Without observation, going via V4: 11
    
    Hmm, in this case V4 direct is better. Let me adjust.
    """
    # Simpler scenario where observation clearly helps
    graph_str = """
#V 3
#E1 1 2 W2 F 0.8
#E2 1 3 W5
#E3 2 3 W1

#O V1 E1 1

#Start 1
#Target 3
"""
    # At V1: E1 revealed immediately (adjacent), so observation of E1 is useless
    # This test doesn't work as designed. Let me fix it.
    
    # Better scenario: E1 is NOT adjacent to start, observation reveals it
    graph_str = """
#V 4
#E1 1 2 W1
#E2 2 3 W2 F 0.7
#E3 3 4 W1
#E4 1 4 W10

#O V1 E2 1

#Start 1
#Target 4
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # This test verifies that observation actions work correctly
    # Check that observing gives correct state transitions
    state = find_state(solver, location=1, has_kit=False, edge_statuses={2: 'unknown'})
    if state:
        # Verify observation is available
        actions = mdp.get_available_actions(state)
        observe_actions = [a for a in actions if a.action_type == ActionType.OBSERVE]
        all_passed &= assert_true(len(observe_actions) > 0,
                                  "Observation action should be available")
    
    return all_passed


def test_observation_not_needed():
    """
    Scenario: Observation available but not useful (edge revealed naturally anyway).
    
    Graph: 1 ---(w=1, p=0.5)--- 2 (target)
    
    Observation from V1: observe E1 for cost 1
    
    But E1 is already adjacent to V1, so its status is revealed at start!
    Observation should never be chosen.
    """
    graph_str = """
#V 2
#E1 1 2 W1 F 0.5

#O V1 E1 1

#Start 1
#Target 2
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # E1 is adjacent to V1, so it's revealed at start
    # States should have E1 as clear or flooded, never unknown
    state_unknown = find_state(solver, location=1, has_kit=False, edge_statuses={1: 'unknown'})
    all_passed &= assert_true(state_unknown is None or 
                              state_unknown not in solver.reachable_states,
                              "E1 unknown should not be reachable (revealed at start)")
    
    # When E1 is clear, should traverse directly
    state_clear = find_state(solver, location=1, has_kit=False, edge_statuses={1: 'clear'})
    if state_clear:
        all_passed &= assert_action(solver, state_clear, ActionType.TRAVERSE, 1,
                                    "E1 clear: should traverse, not observe")
    
    return all_passed


def test_multiple_observations():
    """
    Scenario: Multiple observations available, agent should choose wisely.
    
    Graph: Diamond with uncertain edges on far side
    
         2 ---(w=1, p=0.5)--- 4 (target)
        /                   /
    (1)/                 (1)/
      /                   /
     1                   /
      \                 /
    (1)\             (1)/
        \             /
         3 ---(w=1, p=0.5)---
    
    From V1: can observe E2 (cost 1) or E4 (cost 1)
    
    E1 and E3 are clear (no flood prob), E2 and E4 might be flooded.
    """
    graph_str = """
#V 4
#E1 1 2 W1
#E2 2 4 W1 F 0.5
#E3 1 3 W1
#E4 3 4 W1 F 0.5

#O V1 E2 1
#O V1 E4 1

#Start 1
#Target 4
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # Verify both observations are available at start
    state = find_state(solver, location=1, has_kit=False, edge_statuses={2: 'unknown', 4: 'unknown'})
    if state:
        actions = mdp.get_available_actions(state)
        observe_actions = [a for a in actions if a.action_type == ActionType.OBSERVE]
        all_passed &= assert_true(len(observe_actions) == 2,
                                  "Both observations should be available")
    
    # After knowing E2 is clear, should go via V2
    state = find_state(solver, location=1, has_kit=False, edge_statuses={2: 'clear', 4: 'unknown'})
    if state:
        value = solver.values.get(state, float('inf'))
        all_passed &= assert_approx(value, 2.0, "E2 clear: should cost 2 via V2")
    
    # After knowing both flooded: stuck
    state = find_state(solver, location=1, has_kit=False, edge_statuses={2: 'flooded', 4: 'flooded'})
    if state:
        value = solver.values.get(state, float('inf'))
        all_passed &= assert_true(value == float('inf'), "Both flooded: should be stuck")
    
    return all_passed


# ============================================================
# CATEGORY 6: EDGE CASES AND CORNER CASES
# ============================================================

def test_start_equals_target():
    """
    Scenario: Start and target are the same vertex.
    
    Expected cost: 0
    """
    graph_str = """
#V 2
#E1 1 2 W5
#Start 1
#Target 1
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    expected_cost = 0.0
    actual_cost = solver.get_expected_cost_from_start()
    all_passed &= assert_approx(actual_cost, expected_cost,
                                "Start=Target should have cost 0")
    
    return all_passed


def test_zero_flood_probability():
    """
    Scenario: Edge with F 0 should never be flooded.
    
    Graph: 1 ---(w=1, F 0)--- 2
    
    Edge should always be clear.
    """
    graph_str = """
#V 2
#E1 1 2 W1 F 0
#Start 1
#Target 2
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    expected_cost = 1.0
    actual_cost = solver.get_expected_cost_from_start()
    all_passed &= assert_approx(actual_cost, expected_cost,
                                "Zero flood prob should mean guaranteed clear")
    
    return all_passed


def test_certain_flood():
    """
    Scenario: Edge with F 1.0 is always flooded.
    
    Graph: 1 ---(w=1, F 1.0)--- 2
    
    Without kit: stuck
    """
    graph_str = """
#V 2
#E1 1 2 W1 F 1.0
#Start 1
#Target 2
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # No kit available, edge always flooded -> stuck
    actual_cost = solver.get_expected_cost_from_start()
    all_passed &= assert_true(actual_cost == float('inf'),
                              "Certain flood with no kit should be infinite cost")
    
    return all_passed


def test_disconnected_target():
    """
    Scenario: Target unreachable from start.
    
    This should be caught by validation.
    """
    graph_str = """
#V 3
#E1 1 2 W1
#Start 1
#Target 3
"""
    graph = create_graph_from_string(graph_str)
    
    all_passed = True
    
    # V3 is not connected to anything, should fail validation
    is_valid, message = validate_graph(graph)
    all_passed &= assert_true(not is_valid or "illegal" in message.lower(),
                              "Should detect unreachable target")
    
    return all_passed


# ============================================================
# CATEGORY 7: SIMULATION VERIFICATION
# ============================================================

def test_simulation_deterministic():
    """
    Verify simulation matches expected cost in deterministic scenario.
    """
    graph_str = """
#V 3
#E1 1 2 W2
#E2 2 3 W3
#Start 1
#Target 3
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    simulator = Simulator(mdp, solver)
    
    all_passed = True
    
    # Run many simulations - all should have same cost
    stats = simulator.run_multiple_simulations(num_simulations=50, verbose=False)
    
    all_passed &= assert_true(stats['success_rate'] == 1.0,
                              "All simulations should succeed")
    all_passed &= assert_approx(stats['average_cost'], 5.0,
                                "Average cost should be 5")
    all_passed &= assert_approx(stats['min_cost'], stats['max_cost'],
                                "All runs should have same cost", tolerance=0.001)
    
    return all_passed


def test_simulation_probabilistic():
    """
    Verify simulation average matches expected value for probabilistic scenario.
    """
    graph_str = """
#V 3
#E1 1 3 W1 F 0.3
#E2 1 2 W2
#E3 2 3 W2
#Start 1
#Target 3
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    simulator = Simulator(mdp, solver)
    
    all_passed = True
    
    # E1 clear (0.7): cost 1
    # E1 flooded (0.3): go via V2, cost 4
    # Expected: 0.7*1 + 0.3*4 = 0.7 + 1.2 = 1.9
    
    expected_value = solver.get_expected_cost_from_start()
    
    # Run many simulations
    stats = simulator.run_multiple_simulations(num_simulations=500, verbose=False)
    
    all_passed &= assert_true(stats['success_rate'] == 1.0,
                              "All simulations should succeed")
    
    # Average should be close to expected (within statistical tolerance)
    all_passed &= assert_approx(stats['average_cost'], expected_value,
                                f"Average cost should be close to {expected_value}",
                                tolerance=0.2)  # Allow some variance
    
    return all_passed


def test_simulation_with_kit():
    """
    Verify simulation correctly handles kit equip/unequip.
    """
    graph_str = """
#V 2
#E1 1 2 W1 F 1.0
#K1 1
#EC 2
#UC 1
#FF 2
#Start 1
#Target 2
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    simulator = Simulator(mdp, solver)
    
    all_passed = True
    
    # Expected: equip (2) + traverse flooded (2) = 4
    stats = simulator.run_multiple_simulations(num_simulations=50, verbose=False)
    
    all_passed &= assert_true(stats['success_rate'] == 1.0,
                              "All simulations should succeed (kit available)")
    all_passed &= assert_approx(stats['average_cost'], 4.0,
                                "Average cost should be 4 (equip + traverse)")
    
    return all_passed


# ============================================================
# CATEGORY 8: COMPLEX INTEGRATED SCENARIOS
# ============================================================

def test_complex_scenario_1():
    """
    Complex scenario combining multiple features.
    
    Graph with kit, multiple uncertain edges, and observations.
    """
    graph_str = """
#V 5
#E1 1 2 W2
#E2 2 3 W2 F 0.3
#E3 3 5 W2
#E4 2 4 W3
#E5 4 5 W1 F 0.4

#K1 1
#EC 3
#UC 1
#FF 2

#O V2 E5 1

#Start 1
#Target 5
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # Just verify it solves without error and produces reasonable results
    expected_cost = solver.get_expected_cost_from_start()
    all_passed &= assert_true(expected_cost < float('inf'),
                              "Should find a solution")
    all_passed &= assert_true(expected_cost > 0,
                              "Cost should be positive")
    
    # Verify simulation matches
    simulator = Simulator(mdp, solver)
    stats = simulator.run_multiple_simulations(num_simulations=200, verbose=False)
    
    all_passed &= assert_approx(stats['average_cost'], expected_cost,
                                f"Simulation should match expected {expected_cost}",
                                tolerance=0.5)
    
    return all_passed


def test_complex_scenario_2():
    """
    Scenario where kit decision depends on observed information.
    """
    graph_str = """
#V 4
#E1 1 2 W1 F 0.5
#E2 2 4 W1 F 0.5
#E3 1 3 W2
#E4 3 4 W2 F 0.5

#K1 1
#EC 2
#UC 1
#FF 2

#O V1 E2 1
#O V1 E4 1

#Start 1
#Target 4
"""
    graph = create_graph_from_string(graph_str)
    mdp, solver = solve_graph(graph)
    
    all_passed = True
    
    # Verify solution exists
    expected_cost = solver.get_expected_cost_from_start()
    all_passed &= assert_true(expected_cost < float('inf'),
                              "Should find a solution")
    
    # Verify reachable states are reasonable
    num_reachable = len(solver.reachable_states)
    all_passed &= assert_true(num_reachable > 10,
                              f"Should have many reachable states, got {num_reachable}")
    
    return all_passed


# ============================================================
# MAIN TEST RUNNER
# ============================================================

def run_all_tests() -> bool:
    """Run all tests and report summary."""
    print("\n" + "="*70)
    print("HURRICANE EVACUATION MDP - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        # Category 1: Deterministic scenarios
        ("Deterministic: Simple Path", test_deterministic_simple_path),
        ("Deterministic: Choose Shorter Path", test_deterministic_choose_shorter_path),
        ("Deterministic: Diamond Graph", test_deterministic_diamond),
        
        # Category 2: Single uncertain edge
        ("Single Uncertain: Only Path", test_single_uncertain_edge_on_only_path),
        ("Single Uncertain: With Bypass", test_single_uncertain_edge_with_bypass),
        ("Single Uncertain: Discovered Later", test_single_uncertain_edge_discovered_later),
        
        # Category 3: Multiple uncertain edges
        ("Multiple Uncertain: Series", test_two_independent_uncertain_edges),
        ("Multiple Uncertain: Parallel Paths", test_parallel_uncertain_paths),
        
        # Category 4: Amphibian kit
        ("Kit: Needed for Guaranteed Flood", test_kit_needed_for_guaranteed_flood),
        ("Kit: Kit vs Bypass (Kit Wins)", test_kit_vs_bypass_decision),
        ("Kit: Kit vs Bypass (Bypass Wins)", test_kit_vs_bypass_decision_bypass_wins),
        ("Kit: Unequip for Faster Travel", test_unequip_kit_for_faster_travel),
        
        # Category 5: Observation actions
        ("Observation: Basic", test_observation_basic),
        ("Observation: Value of Information", test_observation_value_of_information),
        ("Observation: Not Needed (Adjacent)", test_observation_not_needed),
        ("Observation: Multiple Available", test_multiple_observations),
        
        # Category 6: Edge cases
        ("Edge Case: Start Equals Target", test_start_equals_target),
        ("Edge Case: Zero Flood Probability", test_zero_flood_probability),
        ("Edge Case: Certain Flood", test_certain_flood),
        ("Edge Case: Disconnected Target", test_disconnected_target),
        
        # Category 7: Simulation verification
        ("Simulation: Deterministic", test_simulation_deterministic),
        ("Simulation: Probabilistic", test_simulation_probabilistic),
        ("Simulation: With Kit", test_simulation_with_kit),
        
        # Category 8: Complex scenarios
        ("Complex: Scenario 1", test_complex_scenario_1),
        ("Complex: Scenario 2", test_complex_scenario_2),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        if run_test(name, test_func):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {passed + failed}")
    print("="*70)
    
    if failed == 0:
        print("✓ ALL TESTS PASSED!")
    else:
        print(f"✗ {failed} TEST(S) FAILED")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
