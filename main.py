import sys
import os
from typing import Optional

from parser import parse_file, print_graph_data, validate_graph, GraphData
from belief_state import BeliefStateSpace
from mdp import HurricaneMDP
from value_iteration import ValueIterationSolver, print_state_values
from simulator import Simulator, print_world_state


class HurricaneEvacuationSolver:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.graph: Optional[GraphData] = None
        self.mdp: Optional[HurricaneMDP] = None
        self.solver: Optional[ValueIterationSolver] = None
        self.simulator: Optional[Simulator] = None
    
    def load_and_validate(self) -> bool:
        print("\n" + "=" * 60)
        print("LOADING PROBLEM SPECIFICATION")
        print("=" * 60)
        
        try:
            self.graph = parse_file(self.file_path)
            print(f"Successfully parsed: {self.file_path}")
        except FileNotFoundError:
            print(f"Error: File not found: {self.file_path}")
            return False
        except Exception as e:
            print(f"Error parsing file: {e}")
            return False
        
        # Display parsed data
        print_graph_data(self.graph)
        
        # Validate the graph
        is_valid, message = validate_graph(self.graph)
        if not is_valid:
            print(f"\nValidation FAILED: {message}")
            return False
        print(f"\nValidation: {message}")
        
        return True
    
    def build_mdp(self) -> bool:
        if self.graph is None:
            print("Error: Graph not loaded")
            return False
        
        print("\n" + "=" * 60)
        print("CONSTRUCTING BELIEF-STATE MDP")
        print("=" * 60)
        
        try:
            self.mdp = HurricaneMDP(self.graph)
            self.mdp.print_mdp_info()
        except Exception as e:
            print(f"Error constructing MDP: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def solve(self, verbose: bool = True) -> bool:
        if self.mdp is None:
            print("Error: MDP not constructed")
            return False
        
        try:
            self.solver = ValueIterationSolver(self.mdp)
            self.solver.solve(verbose=verbose)
            self.simulator = Simulator(self.mdp, self.solver)
        except Exception as e:
            print(f"Error during value iteration: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def print_full_policy(self):
        if self.solver is None:
            print("Error: Problem not solved yet")
            return
        
        self.solver.print_policy(full=True)
    
    def print_reachable_policy(self):
        if self.solver is None:
            print("Error: Problem not solved yet")
            return
        
        self.solver.print_policy(full=False)
    
    def print_state_values_grouped(self):
        if self.solver is None:
            print("Error: Problem not solved yet")
            return
        
        print_state_values(self.solver, group_by_location=True)
    
    def run_single_simulation(self, seed: Optional[int] = None, verbose: bool = True):
        if self.simulator is None:
            print("Error: Problem not solved yet")
            return
        
        world = self.simulator.generate_world(seed=seed)
        print_world_state(world, self.graph)
        
        steps, cost, success = self.simulator.run_simulation(
            world=world, verbose=verbose
        )
        
        if success:
            print(f"\n*** SUCCESS - Reached target in cost {cost} ***")
        else:
            print(f"\n*** FAILED - Did not reach target ***")
        
        return steps, cost, success
    
    def run_multiple_simulations(self, num_simulations: int = 100):
        if self.simulator is None:
            print("Error: Problem not solved yet")
            return
        
        print(f"\nRunning {num_simulations} simulations...")
        stats = self.simulator.run_multiple_simulations(
            num_simulations=num_simulations, verbose=False
        )
        
        print("\n" + "=" * 60)
        print(f"SIMULATION STATISTICS (n={num_simulations})")
        print("=" * 60)
        print(f"  Successes: {stats['successes']}")
        print(f"  Failures: {stats['failures']}")
        print(f"  Success rate: {stats['success_rate']*100:.1f}%")
        print(f"  Average cost (successful runs): {stats['average_cost']:.2f}")
        print(f"  Min cost: {stats['min_cost']:.2f}")
        print(f"  Max cost: {stats['max_cost']:.2f}")
        
        # Compare to expected cost from value iteration
        expected = self.solver.get_expected_cost_from_start()
        print(f"\n  Expected cost (from value iteration): {expected:.4f}")
        print(f"  Empirical average cost: {stats['average_cost']:.4f}")


def interactive_session(solver_instance: HurricaneEvacuationSolver):
    print("\n" + "=" * 60)
    print("INTERACTIVE SESSION")
    print("=" * 60)
    print_help()
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if not cmd:
                continue
            
            if cmd in ['q', 'quit', 'exit']:
                print("Goodbye!")
                break
            
            elif cmd in ['h', 'help']:
                print_help()
            
            elif cmd in ['p', 'policy']:
                solver_instance.print_reachable_policy()
            
            elif cmd in ['pa', 'policy_all']:
                solver_instance.print_full_policy()
            
            elif cmd in ['v', 'values']:
                solver_instance.print_state_values_grouped()
            
            elif cmd in ['s', 'sim', 'simulate']:
                solver_instance.run_single_simulation(verbose=True)
            
            elif cmd.startswith('sim '):
                # Simulation with specific seed
                try:
                    seed = int(cmd.split()[1])
                    solver_instance.run_single_simulation(seed=seed, verbose=True)
                except (ValueError, IndexError):
                    print("Usage: sim <seed>")
            
            elif cmd in ['m', 'multi']:
                solver_instance.run_multiple_simulations(100)
            
            elif cmd.startswith('multi '):
                # Multiple simulations with custom count
                try:
                    n = int(cmd.split()[1])
                    solver_instance.run_multiple_simulations(n)
                except (ValueError, IndexError):
                    print("Usage: multi <count>")
            
            elif cmd in ['g', 'graph']:
                print_graph_data(solver_instance.graph)
            
            elif cmd in ['mdp']:
                solver_instance.mdp.print_mdp_info()
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def print_help():
    print("""
Available Commands:
-------------------
  help, h          - Show this help message
  quit, q          - Exit the program
  
Policy & Values:
  policy, p        - Show policy for reachable states
  policy_all, pa   - Show policy for ALL states
  values, v        - Show state values grouped by location
  
Simulation:
  sim, s           - Run a single simulation (random flooding)
  sim <seed>       - Run simulation with specific random seed
  multi, m         - Run 100 simulations and show statistics
  multi <n>        - Run n simulations
  
Display:
  graph, g         - Show graph structure
  mdp              - Show MDP information
""")


def main():
    print("=" * 60)
    print("HURRICANE EVACUATION - BELIEF-STATE MDP SOLVER")
    print("Assignment 4: Decision-Making Under Uncertainty")
    print("=" * 60)
    
    # Get input file
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        print("\nAvailable test files:")
        # List any .txt files in current directory
        txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
        for f in txt_files:
            print(f"  - {f}")
        
        file_path = input("\nEnter input file path: ").strip()
        if not file_path:
            print("No file specified. Creating example input...")
            file_path = create_example_input()
    
    # Create solver instance
    solver_instance = HurricaneEvacuationSolver(file_path)
    
    # Load and validate
    if not solver_instance.load_and_validate():
        sys.exit(1)
    
    # Build MDP
    if not solver_instance.build_mdp():
        sys.exit(1)
    
    # Solve
    if not solver_instance.solve(verbose=True):
        sys.exit(1)
    
    # Print initial results
    solver_instance.print_reachable_policy()
    
    # Run interactive session
    interactive_session(solver_instance)


def create_example_input() -> str:
    example_content = """
; Example input for Hurricane Evacuation MDP
; Small graph with 2 uncertain edges and observation capability

#V 5    ; number of vertices n in graph (from 1 to n)

#E1 1 2 W3       ; Edge from vertex 1 to vertex 2, weight 3
#E2 2 3 W2 F 0.3 ; Edge from vertex 2 to vertex 3, weight 2, flood prob 0.3
#E3 3 4 W3 F 0.4 ; Edge from vertex 3 to vertex 4, weight 3, flood prob 0.4
#E4 4 5 W1       ; Edge from vertex 4 to vertex 5, weight 1
#E5 2 4 W4       ; Edge from vertex 2 to vertex 4, weight 4 (bypass)

#K1 1            ; Amphibian kit at vertex 1
#EC 2            ; Takes 2 units of time to equip
#UC 1            ; Takes 1 unit of time to unequip
#FF 2            ; Movement with kit is 2x slower

; Observation capability (additional requirement)
#O V2 E3 1       ; From V2, can observe E3 for cost 1

#Start 1
#Target 5
"""
    
    filename = "example_input.txt"
    with open(filename, 'w') as f:
        f.write(example_content)
    
    print(f"Created example input file: {filename}")
    return filename


if __name__ == "__main__":
    main()
