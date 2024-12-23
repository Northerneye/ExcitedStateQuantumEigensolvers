import numpy as np
import qiskit
import qiskit_algorithms
from qiskit.quantum_info import SparsePauliOp

def VQE(H, shots=1024):
    estimator = qiskit.primitives.Estimator(options={"shots": shots})
    optimizer = qiskit_algorithms.optimizers.COBYLA()

    # Define the ansatz which will be used to prepare the ground state
    ansatz = qiskit.circuit.library.TwoLocal(2, ["rx", "ry"], "cz", reps=1)
    
    # Run VQE and get results
    vqe = qiskit_algorithms.VQE(estimator, ansatz, optimizer)
    result = vqe.compute_minimum_eigenvalue(operator=H)
    return result.eigenvalue

if __name__ == "__main__":
    H = SparsePauliOp.from_list([("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5),])

    energy = VQE(H)
    print("Energy: "+str(energy))