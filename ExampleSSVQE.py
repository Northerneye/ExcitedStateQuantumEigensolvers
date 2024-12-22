import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.circuit import QuantumCircuit
from scipy.optimize import minimize

estimator = Estimator()
H = SparsePauliOp.from_list([("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5),])
ansatz = TwoLocal(2, ["rx", "ry"], "cz", reps=1)

def SSVQE(params, ansatz, hamiltonian, estimator, get_energy=False):
    loss = 0
    weights = [1, 0.4] # Weights to create joint loss function
    energies = [] #List to store the energies found
    k=2 # Number of desired eigenstates
    for i in range(k):
        qc = QuantumCircuit(2)
        qc.x(i) # Need orthogonal initial states for SSVQE
        qc = qc.compose(ansatz.assign_parameters(params)) # Apply the unitary ansatz
        result = estimator.run(qc, hamiltonian).result()
        loss += weights[i]*result.values[0]
        energies.append(result.values[0])
    if(get_energy):
        print(f"Energy: {energies}]")
        return energies
    else:
        print(f"Current Loss: {loss}")
        return loss
    
x0 = 2 * np.pi * np.random.random(ansatz.num_parameters) # Random Initial Parameters

res = minimize(SSVQE, x0, args=(ansatz, H, estimator), method="cobyla")
print(SSVQE(res.x, ansatz, H, estimator, get_energy=True))