import numpy as np
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import TimeEvolutionProblem, VarQITE
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple

# The Hamiltonian of interest
H = SparsePauliOp.from_list([("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5),])

# QIPA can be viewed as a modification to the Hamiltonian
tau = 0.1
H_QIPA = H - tau*(H&H).simplify() + (tau**2)/2*(H&H&H).simplify()

# Initialize required Qiskit objects
var_principle = ImaginaryMcLachlanPrinciple()
estimator = qiskit.primitives.Estimator(options={"shots": 1024})
total_time = 5.0

# Append the tunable parameters
ansatz = qiskit.circuit.library.TwoLocal(2, ["rx", "ry"], "cz", reps=1)
params = np.random.random(ansatz.num_parameters)

# Define and Run the Time Evolution Problem
evolution_problem = TimeEvolutionProblem(H_QIPA, total_time, aux_operators=[H])
qrte = VarQITE(ansatz, params[:], var_principle, estimator, num_timesteps=total_time/tau)
result = qrte.evolve(evolution_problem).observables[-1][0][0]
    
print("Energy: "+str(result))