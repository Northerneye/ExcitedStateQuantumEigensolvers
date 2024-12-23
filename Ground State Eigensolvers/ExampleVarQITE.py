import numpy as np
import qiskit
from qiskit.quantum_info import SparsePauliOp

def VarQITE(H, total_time=5.0):
    from qiskit_algorithms import TimeEvolutionProblem, VarQITE
    from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
    
    var_principle = ImaginaryMcLachlanPrinciple()
    estimator = qiskit.primitives.Estimator(options={"shots": 1024})

    # Append the tunable parameters
    ansatz = qiskit.circuit.library.TwoLocal(2, ["rx", "ry"], "cz", reps=1)
    params = np.random.random(ansatz.num_parameters)

    # Define and Run the Time Evolution Problem
    evolution_problem = TimeEvolutionProblem(H, total_time, aux_operators=[H])
    qrte = VarQITE(ansatz, params[:], var_principle, estimator, num_timesteps=50)
    result = qrte.evolve(evolution_problem).observables[-1][0][0]
    
    return result


H = SparsePauliOp.from_list([("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5),])

energy = VarQITE(H)
print("Energy: "+str(energy))