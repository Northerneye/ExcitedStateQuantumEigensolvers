import numpy as np
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit import QuantumCircuit

import math
import matplotlib.pyplot as plt

#H = SparsePauliOp.from_list([ ("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5),])
H = SparsePauliOp.from_list([("Z", 0.5)])


def get_energy(qc, H):
    estimator = Estimator(options={"shots": 2**15})
    result = estimator.run(qc, H).result()
    return result.values[0]

""" # As described in the paper
def initialize_ancilla(qc):
    return qc

def measure_paulis(qc, H):
    for i in range(len(H.paulis)):
        # Get current pauli string of the Hamiltonian
        pauli = str(H.paulis[i])
        
        # Disgard the Identity string
        if(pauli == "I"*len(pauli)):
            continue
        
        # Find the Binary representation for ancilla control operation
        binary = str(bin(i)[2:])
        while len(binary) < math.ceil(np.log2(len(H.paulis))):
            binary = "0" + binary
        
        # Prepare ancilla for controlled operation
        for j in range(len(binary)):
            if(binary[j] == "0"):
                qc.x(j)

        # Create a gate for the specific pauli string (from Hamiltonian)
        pauli_qc = QuantumCircuit(len(pauli))
        for j in range(len(pauli)):
            if(pauli[j] == "X"):
                pauli_qc.x(j)
            if(pauli[j] == "Y"):
                pauli_qc.y(j)
            if(pauli[j] == "Z"):
                pauli_qc.z(j)
        
        # Add this controlled operation into the circuit
        pauli_qc = pauli_qc.control(math.ceil(np.log2(len(H.paulis))))
        qc.compose(pauli_qc, [i for i in range(qc.num_qubits)], inplace=True)

        # Correct the ancillas back for the next pauli string
        for j in range(len(binary)):
            if(binary[j] == "0"):
                qc.x(j)
        

def final_hadamard(qc, num_ancilla=1):
    for i in range(num_ancilla):
        qc.h(i)

def ensure_zeros(qc):
    return qc

def FQE(H, rounds=1):
    # Normalize coefficients of Hamiltonian
    C = np.sqrt(np.sum(np.abs(H.coeffs)**2))
    H.coeffs = H.coeffs/C
    
    qc = QuantumCircuit(H.num_qubits + math.ceil(np.log2(len(H.paulis))))
    for i in range(rounds):

        # Need to first initialize the ancilla to encode the coefficients of the Hamiltonian
        initialize_ancilla(qc)
        
        # Next we need to perform controlled measurements from the ancilla to the quantum state
        measure_paulis(qc, H)

        # Lastly we need to perform final hadamard gates and ensure that the measurement is zero 
        # (reuse ancilla? - measure to different classical bits and reset ancilla qubits)
        final_hadamard(qc, num_ancilla=math.ceil(np.log2(len(H.paulis))))
        ensure_zeros(qc)
        #input(qc)

    # Run the created circuit and record results
    # (use a sampler to condition the results on measuring all zeros from the ancilla?)

    return True

""" # With single ancilla and resets
def measure_paulis(qc, H):
    global current_classical_bit
    for i in range(len(H.paulis)):
        # Get current pauli string of the Hamiltonian
        pauli = str(H.paulis[i])
        
        # Reset the ancilla qubit
        qc.reset(0)

        # Create a gate for the specific pauli string (from Hamiltonian)
        pauli_qc = QuantumCircuit(len(pauli))
        for j in range(len(pauli)):
            if(pauli[j] == "X"):
                pauli_qc.x(j)
            if(pauli[j] == "Y"):
                pauli_qc.y(j)
            if(pauli[j] == "Z"):
                pauli_qc.z(j)
        
        # Add this controlled operation into the circuit
        pauli_qc = pauli_qc.control(1)
        qc.ry(float(H.coeffs[i]), 0)
        qc.compose(pauli_qc, [i for i in range(qc.num_qubits)], inplace=True)
        
        #qc.h(0)
        qc.ry(-float(H.coeffs[i]), 0)
        
        #qc.measure(0, current_classical_bit)
        current_classical_bit += 1
        #input(qc.decompose())
        

def final_hadamard(qc, num_ancilla=1):
    for i in range(num_ancilla):
        qc.h(i)

def ensure_zeros(qc):
    return qc

def FQE(H, rounds=1):
    # Construct the quantum gradient descent operator
    gamma = 0.01
    I = SparsePauliOp.from_list([("I"*H.num_qubits, 1.0)])
    Hg = I - gamma*H

    # Normalize coefficients of Hamiltonian
    C = np.sqrt(np.sum(np.abs(Hg.coeffs)**2))
    Hg.coeffs = Hg.coeffs/C

    global current_classical_bit
    current_classical_bit = 0
    
    qc = QuantumCircuit(Hg.num_qubits + 1, rounds*len(Hg.coeffs))
    qc.rx(np.pi*3/4, 1)
    for i in range(rounds):

        # Next we need to perform controlled measurements from the ancilla to the quantum state
        measure_paulis(qc, Hg)

    # Run the created circuit and record results
    # (use a sampler to condition the results on measuring all zeros from the ancilla?)
    Single_Identity = SparsePauliOp.from_list([("I", 1.0)])
    expanded_H = Single_Identity.expand(H)
    #input(qc)
    energy = get_energy(qc, expanded_H)

    return energy
#"""

energy = FQE(H)

input("FQE Energy: "+str(energy))