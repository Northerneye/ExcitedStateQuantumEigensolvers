import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.circuit import QuantumCircuit
from scipy.optimize import minimize

def VarianceVQE_round(params, ansatz, hamiltonian, k=2, get_energy=False):
    """
    Runs a single round of VarianceVQE and returns the variance-based loss of the given ansatz and parameters.  
    The variance-based loss is zero when an eigenvector is found.
    Setting get_energy=True returns the energies of the ground and excited states in a list.

    Args:
        params (numpy array): An array of the current parameter values for the ansatz
        ansatz (QuantumCircuit): A parameterized quantum circuit used to estimate the ground and excited state wavefunctions
        hamiltonian (SparsePauliOp): The Hamiltonian operator whose lowest eigenvalues we are interested in.
        get_energy (bool, optional): A boolean used to return the discovered energy eigenvalues (list), rather than the loss (float). Defaults to False.

    Returns:
        if(get_energy==False):
            float: The current variance-based loss from the current round of VaranceVQE
        if(get_energy==True):
            List: The energy eigenvalues found from the provided ansatz and parameters
    """
    estimator = Estimator()
    loss = 0.0
    
    qc = QuantumCircuit(2)
    qc = qc.compose(ansatz.assign_parameters(params)) # Apply the unitary ansatz

    # Variance Loss is <H^2> - <H>^2
    # Find contribution to loss from <H^2>
    result = estimator.run(qc, (hamiltonian&hamiltonian).simplify()).result() 
    loss += result.values[0]

    # Find contribution to loss from <H>^2
    result = estimator.run(qc, hamiltonian).result() 
    loss -= result.values[0]**2

    if(get_energy):
        return result.values[0]
    else:
        return loss
    
def VarianceVQE(H, ansatz, k=2):
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

    # Run VarianceVQE
    res = minimize(VarianceVQE_round, x0, args=(ansatz, H, k), method="cobyla")
    return VarianceVQE_round(res.x, ansatz, H, k=k, get_energy=True)


if __name__ == "__main__":
    
    # Define the system Hamiltonian
    H = SparsePauliOp.from_list([("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5),])

    # Define the circuit ansatz
    ansatz = TwoLocal(2, ["rx", "ry"], "cz", reps=1)
    
    # Run VarianceVQE and print the results
    energies = VarianceVQE(H, ansatz)
    print("Energy: "+str(energies))