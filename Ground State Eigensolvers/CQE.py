import numpy as np
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit import QuantumCircuit

from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

import math
import matplotlib.pyplot as plt

# The Hamiltonian of interest
#H = SparsePauliOp.from_list([("ZI", 1.0)])
H = SparsePauliOp.from_list([ ("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5),])
#H = SparsePauliOp.from_list([("Z", 1.0)])

#mapper = JordanWignerMapper()
#fermionic_op = FermionicOp({"+_0 +_1": 1.0}, num_spin_orbitals=2)
#H = mapper.map(fermionic_op)

def get_energy(qc, H):
    estimator = Estimator(options={"shots": 2**15})
    result = estimator.run(qc, H).result()
    return result.values[0]

""" # Apply pauli rotation (For Scalability) 
def apply_pauli(qc, pauli, theta):
    qc_a = QuantumCircuit(qc.num_qubits + 1)
    qc_a.compose(qc, [i for i in range(qc.num_qubits)], inplace=True)

    for i in range(len(pauli)):
        if(pauli[i] == "X"):
            qc_a.h(i)
        elif(pauli[i] == "Y"):
            qc_a.rx(np.pi/2, i)

        if(pauli[i] != "I"):
            qc_a.cx(i, qc.num_qubits)
    
    qc_a.rz(theta, qc.num_qubits)

    for i in reversed(range(len(pauli))):
        if(pauli[i] != "I"):
            qc_a.cx(i, qc.num_qubits)
            
        if(pauli[i] == "X"):
            qc_a.h(i)
        elif(pauli[i] == "Y"):
            qc_a.rx(-np.pi/2, i)

    input(qc_a)
    return qc_a
"""
#''' # Two Qubit Tunable Pauli Gates
def apply_pauli(qc, pauli, theta):
    if(pauli == "IX"):
        qc.rx(theta, 0)
    elif(pauli == "IY"):
        qc.ry(theta, 0)
    elif(pauli == "IZ"):
        qc.rz(theta, 0)
        
    if(pauli == "XI"):
        qc.rx(theta, 1)
    elif(pauli == "YI"):
        qc.ry(theta, 1)
    elif(pauli == "ZI"):
        qc.rz(theta, 1)

    
    if(pauli == "XX"):
        qc.rxx(theta, 0, 1)
    elif(pauli == "YX"):
        qc.h(0)
        qc.h(1)
        qc.s(1)
        qc.rzz(theta, 0, 1)
        qc.sdg(1)
        qc.h(1)
        qc.h(0)
    elif(pauli == "ZX"):
        qc.h(0)
        qc.rzz(theta, 0, 1)
        qc.h(0)
    
    if(pauli == "XY"):
        qc.h(0)
        qc.s(0)
        qc.h(1)
        qc.rzz(theta, 0, 1)
        qc.h(1)
        qc.sdg(0)
        qc.h(0)
    elif(pauli == "YY"):
        qc.ryy(theta, 0, 1)
    elif(pauli == "ZY"):
        qc.h(0)
        qc.s(0)
        qc.rzz(theta, 0, 1)
        qc.sdg(0)
        qc.h(0)
    
    if(pauli == "XZ"):
        qc.h(1)
        qc.rzz(theta, 0, 1)
        qc.h(1)
    elif(pauli == "YZ"):
        qc.h(1)
        qc.s(1)
        qc.rzz(theta, 0, 1)
        qc.sdg(1)
        qc.h(1)
    elif(pauli == "ZZ"):
        qc.rzz(theta, 0, 1)
    
    return qc
''' # Single Qubit Tunable Pauli Gates
def apply_pauli(qc, pauli, theta):
    if(pauli == "X"):
        qc.rx(theta, 0)
    elif(pauli == "Y"):
        qc.ry(theta, 0)
    elif(pauli == "Z"):
        qc.rz(theta, 0)

    return qc
#'''
#"""

def get_A(qc, pauli1, H):
    estimator = Estimator(options={"shots": 2**19})
    total = 0.0
    
    # Multiply sigma_i by H to get the observable
    observables = (pauli1&H).simplify()

    # Estimator doesn't like complex coefficients
    for i in range(len(observables.paulis)):
        coefficient = observables.coeffs[i]
        pauli = SparsePauliOp.from_list([(str(observables.paulis[i]), 1.0)])
        result = estimator.run(qc, pauli).result()
        total += result.values[0]*coefficient

    # A_ijkl = <psi|[a^dag_i a^dag_j a_k a_l, H]|psi>, need to complete the commutator (did the first term in code above)
    observables = (H&pauli1).simplify()
    for i in range(len(observables.paulis)):
        coefficient = observables.coeffs[i]
        pauli = SparsePauliOp.from_list([(str(observables.paulis[i]), 1.0)])
        result = estimator.run(qc, pauli).result()
        total -= result.values[0]*coefficient
    
    return total.real

def Measure_A(qc, H):
    A = np.zeros((H.num_qubits, H.num_qubits, H.num_qubits, H.num_qubits))
    A_op = SparsePauliOp.from_list([("II", 0.0)])
    mapper = JordanWignerMapper()
    for i in range(H.num_qubits):
        for j in range(H.num_qubits):
            for k in range(H.num_qubits):
                for l in range(H.num_qubits):
                    # Get the corresponding pauli operator to this fermionic operator
                    fermionic_op = FermionicOp({"+_"+str(i)+" +_"+str(j)+" -_"+str(k)+" -_"+str(l): 1.0}, num_spin_orbitals=H.num_qubits)
                    qubit_jw_op = mapper.map(fermionic_op)

                    # Construct Circuit to measure A[i,j,k,l] = <psi|[pauli1, H]|psi>
                    A[i,j,k,l] = get_A(qc, qubit_jw_op, H)
                    A_op += A[i,j,k,l]*qubit_jw_op
            
            A_op = A_op.simplify()

    return A, A_op

def CQE(H, steps=5, eps=0.1, tolerance=1e-2):
    # Initialize the Quantum State
    qc = QuantumCircuit(H.num_qubits)
    qc.x(0)
    qc.h(0)
    qc.h(1)
    
    current_time = 0.0
    energies = []
    while(current_time < final_time):
        print("t="+str(current_time))
        input(qc)

        # Acquire the A matrix
        A, A_op = Measure_A(qc, H)

        input(A_op) 
        input(A)

        # Now use the A matrix to apply the next layer of gates
        for i in range(len(A_op.paulis)):
            pauli = A_op.paulis[i]
            coeff = A_op.coeffs[i]
            apply_pauli(qc, pauli, coeff)

        energies.append(get_energy(qc, H))

        current_time += dt
    
    plt.plot(energies)
    plt.title("TrotterQITE")
    plt.xlabel("Timestep")
    plt.ylabel("Energy (eV)")
    plt.show()



final_time = 5.0
dt = 0.5
result = CQE(H, steps=5, eps=0.1)