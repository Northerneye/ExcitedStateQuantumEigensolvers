import numpy as np
from qiskit_algorithms import VarQRTE
from qiskit.circuit.library import ExcitationPreserving
from qiskit_algorithms import TimeEvolutionProblem
from qiskit_algorithms.time_evolvers.variational import RealMcLachlanPrinciple
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

var_principle = RealMcLachlanPrinciple()
estimator = Estimator(options={"shots": 1024})

hamiltonian = SparsePauliOp.from_list([("ZI", 0.5), ("IZ", 0.5), ("XX", 0.2)])

def init_circ():
    qc = QuantumCircuit(2,0)
    qc.x(0) # initial state is |10>
    return qc

# Append the tunable parameters
anstaz = ExcitationPreserving(num_qubits=2, entanglement='linear', reps=1)

# Anstaz at t=0 must be equal to the Identity
params = np.array([0.0 for i in range(5)])

# Combine the initial state and tunable anstaz to create our quantum circuit
anstaz = init_circ().compose(anstaz)

total_time = 5.0
evolution_timestep = 0.2

# Define and Run the Time Evolution Problem
evolution_problem = TimeEvolutionProblem(hamiltonian, total_time)
qrte = VarQRTE(anstaz, params[:], variational_principle=var_principle, estimator=estimator, num_timesteps=int(total_time/evolution_timestep))
params = qrte.evolve(evolution_problem).parameter_values

# Assemble the circuit which creates the evolved state
evolved_circ = anstaz.assign_parameters(params[-1])
print(evolved_circ)