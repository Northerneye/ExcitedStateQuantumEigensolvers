import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.circuit import QuantumCircuit
from scipy.optimize import minimize

def SSVQE_round(params, ansatz, hamiltonian, k=2, get_energy=False):
    """
    Runs a single round of SSVQE and returns the loss of the given ansatz and parameters.  
    Setting get_energy=True returns the energies of the ground and excited states in a list.

    Args:
        params (numpy array): An array of the current parameter values for the ansatz
        ansatz (QuantumCircuit): A parameterized quantum circuit used to estimate the ground and excited state wavefunctions
        hamiltonian (SparsePauliOp): The Hamiltonian operator whose lowest eigenvalues we are interested in.
        get_energy (bool, optional): A boolean used to return the discovered energy eigenvalues (list), rather than the loss (float). Defaults to False.

    Returns:
        if(get_energy==False):
            float: The current weighted loss from the current round of SSVQE
        if(get_energy==True):
            List: The energy eigenvalues found from the provided ansatz and parameters
    """
    estimator = Estimator()
    loss = 0
    weights = [1, 0.4] # Weights to create joint loss function
    energies = [] # List to store the energies found

    # Loop over all desired states
    for i in range(k):
        qc = QuantumCircuit(2)
        qc.x(i) # Need orthogonal initial states for SSVQE
        qc = qc.compose(ansatz.assign_parameters(params)) # Apply the unitary ansatz
        result = estimator.run(qc, hamiltonian).result()
        loss += weights[i]*result.values[0]
        energies.append(result.values[0])

    if(get_energy):
        return energies
    else:
        return loss
    
def SSVQE(H, ansatz, k=2):
    """_summary_

    Args:
        H (SparsePauliOp): The Hamiltonian of the system of interest.
        ansatz (QuantumCircuit): A parameterized Quantum Circuit used to find the ground and excited states of the Hamiltonian
        k (int, optional): The number of eigenvalues to find. Defaults to 2.

    Returns:
        list: A list of the found eigenvalues.
    """
    # Define the ansatz
    x0 = 2 * np.pi * np.random.random(ansatz.num_parameters) # Random Initial Parameters

    # Run SSVQE
    res = minimize(SSVQE_round, x0, args=(ansatz, H, k), method="cobyla")
    return SSVQE_round(res.x, ansatz, H, k=k, get_energy=True)


if __name__ == "__main__":
    
    # Define the system Hamiltonian
    H = SparsePauliOp.from_list([("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5),])

    # Define the circuit ansatz
    ansatz = TwoLocal(2, ["rx", "ry"], "cz", reps=1)
    
    # Run SSVQE and print the results
    energies = SSVQE(H, ansatz)
    print("Energies: "+str(energies))