import numpy as np
from qiskit import QuantumCircuit
import qiskit
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
import math

def apply_param(params, parameter, qc, N=2, start=0):
    """
    The function used to define the ansatz.  Do not need to return anything as these operators are appended to qc.
    THIS MUST BE CHANGED WHEN CHANGING THE ANSATZ.  The terms below define the structure of the ansatz.  Try printing the circuit to see.

    Args:
        params (numpy array): The current parameter values of the ansatz.
        parameter (int): The current parameter index.
        qc (QuantumCircuit): A quantum circuit constructing the ansatz up to the current parameter.
        N (int, optional): The number of qubits in the ansatz. Defaults to 2.
        start (int, optional): The starting qubit index, relevant for the Deflation term swap tests. Defaults to 0.
    """
    if(math.floor(parameter/N)%2 == 0):
        qc.rx(params[parameter], start + parameter%N)
    else:
        qc.ry(params[parameter], start + parameter%N)
    if((parameter+1)%(2*N) == 0 and len(params) > parameter+1):
        for i in range(N-1):
            qc.cx(start + i, start + i + 1)
    
def measure_der(parameter, qc, N=2, start=0):
    """
    A function used to measure the derivative of a parameter in the ansatz.  Do not need to return anything as these operators are appended to qc.
    THIS MUST BE CHANGED WHEN CHANGING THE ANSATZ.  The terms below represent the derivatives of the parameters in apply_param. Ex derivative of RX is X.

    Args:
        parameter (int): The index of the current parameter.
        qc (QuantumCircuit): The quantum circuit up the parameter of interest.
        N (int, optional): The number of qubits in the ansatz. Defaults to 2.
        start (int, optional): The starting index of the quantum circuit (important for the Deflation terms). Defaults to 0.
    """
    if(math.floor(parameter/2)%2 == 0):
        qc.cx(start + N, parameter%2)
    else:
        qc.cy(start + N, parameter%2)

def pauli_measure(qc, pauli_string):
    """
    Perform hadamard tests to measure the pauli string of interest.  Do not need to return anything as these operators are appended to qc.

    Args:
        qc (QuantumCircuit): A quantum circuit which prepares the quantum state of interest.
        pauli_string (str): The current pauli string to be measured.
    """
    N = len(pauli_string)
    for i in range(len(pauli_string)): # Measure Pauli Strings
        if(str(pauli_string[i]) == "X"):
            qc.cx(N,i)
        if(str(pauli_string[i]) == "Y"):
            qc.cy(N,i)
        if(str(pauli_string[i]) == "Z"):
            qc.cz(N,i)
    
def Hamiltonian_Circuit(params, pauli_string):
    """
    A function to generate quantum circuits used to measure the contribution of certain pauli strings to the energy.  Called in the energy() function.

    Args:
        params (numpy array): The current values of the parameters of the ansatz.
        pauli_string (string): The current pauli string being measured.  The coefficient 
                               of this pauli string will be recombined with this result in energy().

    Returns:
        QuantumCircuit: A quantum circuit used to measure the contribution of certain pauli strings to the energy.
    """
    N = len(pauli_string)
    qc = QuantumCircuit(N+1, 1)
    qc.h(N)
    for parameter in range(len(params)): # Apply parameterized gates
        apply_param(params, parameter, qc, N=N)
    pauli_measure(qc, pauli_string)
    qc.h(N)
    return qc

def energy(H, params, shots=2**10):
    """
    A function to find the energy of the current ansatz and parameters on the given Hamiltonian.

    Args:
        H (SparsePauliOp): The Hamiltonian of interest.
        params (numpy array): The current parameters of the ansatz.
        shots (int, optional): The number of shots used to measure observables. Defaults to 2**10.

    Returns:
        float: The energy of the current quantum state.
    """
    E = 0.0
    estimator = Estimator(options={"shots": shots})
    for pauli_string in range(len(H.paulis)):
        qc = Hamiltonian_Circuit(params, H.paulis[pauli_string])
        result = estimator.run(qc, SparsePauliOp("Z"+"I"*H.num_qubits)).result()
        E += H.coeffs[pauli_string].real*result.values[0]
    return E

def A_Circuit(params, i, j, N=2):
    """
    The quantum circuit used to determine the value of elements in the A matrix for QITE.

    Args:
        params (numpy array): The current parameters of the ansatz used to realize the current guess at the quantum eigenstate.
        i (int): The first index of the A matrix entry being determined. 
        j (int): The second index of the A matrix entry being determined. 
        N (int, optional): The number of qubits in the ansatz. Defaults to 2.

    Returns:
        QuantumCircuit: The quantum circuit used to determine the value of A[i,j].
    """
    qc = QuantumCircuit(N+1, 1)
    qc.h(N)
    for parameter in range(len(params)):# Apply parameterized gates
        if(parameter == i):
            qc.x(N)
            measure_der(parameter, qc, N=N) # Measure generators
            qc.x(N)
        if(parameter == j):
            measure_der(parameter, qc, N=N) # Measure second generators
        apply_param(params, parameter, qc, N=N)
    qc.h(N)
    return qc

def Measure_A(params, N=2, shots=2**10):
    """
    A function to determine the A matrix in QITE.  This creates quantum circuits to measure each element of 
    the A matrix by comparing hadamard tests of multiple parameter derivatives.

    Args:
        params (numpy array): The current parameters of the ansatz used to realize the current quantum state.
        N (int, optional): The number of qubits in the ansatz. Defaults to 2.
        shots (int, optional): The number of shots used to determine each observable. Defaults to 2**10.

    Returns:
        list: The A matrix used along with the C vector to determine the next step of QITE.
    """
    observable = SparsePauliOp("Z"+"I"*N)
    estimator = Estimator(options={"shots": shots})
    A = [[0.0 for i in range(len(params))] for j in range(len(params))]
    for i in range(len(params)):
        for j in range(len(params)-i):
            qc = A_Circuit(params, i, i+j, N=N)
            result = estimator.run(qc, observable).result()
            A[i][i+j] = 1/4*result.values[0]
            if(j != 0):
                A[i+j][i] = 1/4*result.values[0]
    return A

def C_Circuit(params, i, pauli_string, N=2):
    """
    The quantum circuits which determine elements of the C vector in QITE.

    Args:
        params (numpy array): The current parameters of the ansatz
        i (int): The index of the component of the C vector being determined.  
                 Used to find which parameter we need to take a derivate of.
        pauli_string (string): The current pauli string of the Hamiltonian which we are computing.  
                               We will need to sum up contributions from all pauli strings in Measure_C
        N (int, optional): The number of qubits in the ansatz. Defaults to 2.

    Returns:
        QuantumCircuit: The quantum circuit to be measured in order to determine a contribution to the i-th element of the C vector in QITE.
    """
    qc = QuantumCircuit(N+1, 1)
    qc.h(N)
    qc.s(N)#To get only imaginary component
    for parameter in range(len(params)): # Apply parameterized gates
        if(parameter == i):
            qc.x(N)
            measure_der(parameter, qc) # Measure generators
            qc.x(N)
        apply_param(params, parameter, qc, N=N)
    pauli_measure(qc, pauli_string)
    qc.h(N)
    return qc

def Measure_C(H, params, all_params, alpha, shots=2**10):
    """
    A function to determine the C vector in QITE.  This creates quantum circuits to measure each element of 
    the C vector and factors in the overlap terms of previously determined eigenstates through the deflation terms.

    Args:
        H (SparsePauliOp): The Hamiltonian of interest.
        params (numpy array): A list of the current parameter values of the anstaz.
        all_param (list): A list of all previously discovered states through their parameter values. 
                          This is used to check the overlap of current states with previous ones.
        alpha (list): A list of floats which contains the coefficients of the deflation terms.  
                      The strength of the repulsion between current and previously found eigenstates.
        shots (int, optional): The number of shots to determine an observable. Defaults to 2**10.

    Returns:
        list: The C vector for use in determining theta_dot in QITE.
    """

    N = H.num_qubits
    C = [0.0 for i in range(len(params))]
    observable = SparsePauliOp("Z"+"I"*N)
    repulsion_observable = SparsePauliOp("Z"+"I"*(2*N))
    estimator = Estimator(options={"shots": shots})
    for i in range(len(params)):

        # Compute the current values of the C vector as if it was the ground state
        for pauli_string in range(len(H.paulis)):
            qc = C_Circuit(params, i, H.paulis[pauli_string])
            result = estimator.run(qc, observable).result()
            C[i] -= 1/2*H.coeffs[pauli_string].real*result.values[0]

        # Check overlap with previous eigenstates
        for level in range(len(all_params)):
            repulsionqc = Deflation_Circuit(params, i, all_params[level], N=H.num_qubits)
            result = estimator.run(repulsionqc, repulsion_observable).result()
            C[i] -= 1/2*alpha[level]*result.values[0]

    return C

def Deflation_Circuit(params, i, all_param, N=2):
    """
    The circuit used to compare the overlap of new eigenstates with previously discovered eigenstates. 
    Used to estimate the added loss due to overlapping with previously found low energy eigenstates.

    Args:
        params (numpy array): The current value of the ansatz parameters/
        i (int): The index of the parameter of interest.  A derivate will be taken with respect to this
                 parameter in order to decrease the overlap of previous states.
        all_param (list): A list of all previously discovered states through their parameter values. 
                          This is used to check the overlap of current states with previous ones.
        N (int, optional): The number of qubits of the ansatz. Defaults to 2.

    Returns:
        QuantumCircuit: The quantum circuit of a swap test of the current eigenstates with previously found lower energy eigenstates.
    """
    qc = QuantumCircuit(2*N+1, 1)
    qc.h(2*N)
    qc.s(2*N) # To get only imaginary component
    for parameter in range(len(params)): # Apply parameterized gates
        if(parameter == i):
            qc.x(2*N)
            measure_der(parameter, qc, start=N)
            qc.x(2*N)
        apply_param(params, parameter, qc, N=N)
    
    # Prepare previous energy level
    for parameter in range(len(all_param)):
        apply_param(all_param, parameter, qc, N=N, start=N)
    
    # Controlled Swap Test
    for qubit in range(N):
        qc.cswap(2*N, qubit, N+qubit)
    qc.h(2*N)
    return qc

def QITED(H, alpha=None, energy_levels=2, max_iter=100, step_size=0.2, shots=2**15):
    """
    Runs the QITED Algorithm.  The twolocal ansatz is hardcoded into the apply_param and measure_der functions.
    Functions run the A and C circuits to generate the A matrix and C vector.  The parameter step is found
    by inverting A and multiplying it by C to generate theta_dot.  This is then used to evolve the state through
    imaginary time to uncover the ground state.

    Args:
        H (SparsePauliOp): The Hamiltonian of interest.
        alpha (list, optional): The increase in energy applied to discovered states, in order to allow the ground state algorithm to identify excited states. Defaults to None.
        energy_levels (int, optional): The number of energy levels to be returned by QITED. Defaults to 2.
        max_iter (int, optional): The maximum number of timesteps the QITED algorithm will take before returning the current state. Defaults to 100.
        step_size (float, optional): The size of each timestep the QITED algorithm takes. Defaults to 0.2.
        shots (int, optional): The number of shots used to estimate observables. Defaults to 2**10.

    Returns:
        list: The discovered energy eigenvalues
    """

    if(alpha is None): # Alpha Increases the energy of already discovered states
        alpha = [5.0 for i in range(energy_levels)]
    
    energies = []
    all_energies = [[] for i in range(energy_levels)]
    all_params = []
    for energy_level in range(energy_levels): # Add the theta_dots for each energy level
        my_params = (2*np.pi*np.random.rand(8)).tolist() # Reset Initial Parameters after each run

        for i in range(max_iter):
            theta_dot = np.array([0.0 for j in range(len(my_params))])
            A = np.array(Measure_A(my_params, N=H.num_qubits, shots=shots))
            C = np.array(Measure_C(H, my_params, all_params, alpha, shots=shots))

            #Approximately invert A using Truncated SVD
            u,s,v=np.linalg.svd(A)
            for j in range(len(s)): 
                if(s[j] < 0.02):
                    s[j] = 1e6
            t = np.diag(s**-1)
            A_inv=np.dot(v.transpose(),np.dot(t,u.transpose()))
            theta_dot = np.matmul(A_inv, C)

            print("Energy Level: "+str(energy_level))
            print("Theta Dot: "+str(np.sum(np.abs(theta_dot))))
            print()

            for j in range(len(theta_dot)):
                my_params[j] += theta_dot[j]*step_size

            all_energies[energy_level].append(energy(H, my_params, shots=shots))

        energies.append(energy(H, my_params, shots=shots))
        all_params.append(my_params[:])

    return energies

if __name__ == "__main__":
    
    # Define the Hamiltonian of interest
    H = SparsePauliOp.from_list([("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5)])

    # Run QITED and print the results
    energies = QITED(H)
    print("QITED Energies: "+str(energies))