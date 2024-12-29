import numpy as np
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator, StatevectorEstimator
from qiskit import QuantumCircuit
import math
import matplotlib.pyplot as plt

# The Hamiltonian of interest
#H = SparsePauliOp.from_list([("ZI", 1.0)])
H = SparsePauliOp.from_list([ ("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5),])
#H = SparsePauliOp.from_list([("Z", 1.0)])

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

def get_M(qc, pauli1, pauli2):
    estimator = Estimator(options={"shots": 2**15})
    
    # Multiply sigma_i by H to get the observable
    sigma_i = SparsePauliOp.from_list([(pauli1, 1.0)])
    sigma_j = SparsePauliOp.from_list([(pauli2, 1.0)])
    observable = (sigma_i&sigma_j).simplify()
    
    # Estimator doesn't like complex coefficients
    observable = SparsePauliOp.from_list([(str(observable.paulis[0]), observable.coeffs[0].real)])
    result = estimator.run(qc, observable).result()
    
    return 2*result.values[0].real

def Measure_M(qc):
    M = np.zeros((4**qc.num_qubits, 4**qc.num_qubits))

    for i in range(4**qc.num_qubits):
        pauli1 = ""
        for j in range(qc.num_qubits):
            if(math.floor(i/4**j)%4 == 0):
                pauli1 += "I"
            elif(math.floor(i/4**j)%4 == 1):
                pauli1 += "X"
            elif(math.floor(i/4**j)%4 == 2):
                pauli1 += "Y"
            elif(math.floor(i/4**j)%4 == 3):
                pauli1 += "Z"
        
        for j in range(4**qc.num_qubits):
            pauli2 = ""
            for k in range(qc.num_qubits):
                if(math.floor(j/4**k)%4 == 0):
                    pauli2 += "I"
                elif(math.floor(j/4**k)%4 == 1):
                    pauli2 += "X"
                elif(math.floor(j/4**k)%4 == 2):
                    pauli2 += "Y"
                elif(math.floor(j/4**k)%4 == 3):
                    pauli2 += "Z"

            # Construct Circuit to measure M[i,j] = <psi|pauli1*pauli2|psi>
            M[i,j] = get_M(qc,pauli1, pauli2)

    return M

def get_B(qc, pauli1, H):
    estimator = Estimator(options={"shots": 2**19})
    total = 0.0
    
    # Multiply sigma_i by H to get the observable
    sigma_i = SparsePauliOp.from_list([(pauli1, 1.0)])
    observables = (sigma_i&H).simplify()

    # Estimator doesn't like complex coefficients
    for i in range(len(observables.paulis)):
        coefficient = observables.coeffs[i]
        pauli = SparsePauliOp.from_list([(str(observables.paulis[i]), 1.0)])
        result = estimator.run(qc, pauli).result()
        total += result.values[0]*coefficient


    # B = Im(<psi|[H,sigma]|psi>), need to complete the commutator (did the first term in code above)
    observables = (H&sigma_i).simplify()
    for i in range(len(observables.paulis)):
        coefficient = observables.coeffs[i]
        pauli = SparsePauliOp.from_list([(str(observables.paulis[i]), 1.0)])
        result = estimator.run(qc, pauli).result()
        total -= result.values[0]*coefficient
    
    return total.imag

def Measure_B(qc, H):
    B = np.zeros((4**qc.num_qubits))
    for i in range(4**qc.num_qubits):
        pauli1 = ""
        for j in range(qc.num_qubits):
            if(math.floor(i/4**j)%4 == 0):
                pauli1 += "I"
            elif(math.floor(i/4**j)%4 == 1):
                pauli1 += "X"
            elif(math.floor(i/4**j)%4 == 2):
                pauli1 += "Y"
            elif(math.floor(i/4**j)%4 == 3):
                pauli1 += "Z"
        
        # Construct Circuit to measure B[i] = <psi|[H,pauli1]|psi>
        B[i] = get_B(qc, pauli1, H)

    return B

def TrotterQITE(H, final_time=1.0, dt=0.1, tolerance=1e-2):
    # Initialize the Quantum State
    qc = QuantumCircuit(H.num_qubits)
    qc.x(0)
    qc.h(0)
    qc.h(1)
    
    current_time = 0.0
    energies = []
    while(current_time < final_time):
        print("t="+str(current_time))
        M = np.array(Measure_M(qc))
        B = np.array(Measure_B(qc, H))

        # Approximately invert A using Truncated SVD
        u,s,v=np.linalg.svd(M)
        for j in range(len(s)): # Make matrix invertible (but still throw out bad value)
            if(s[j] == 0):
                s[j] = 0.00000001
        t = np.diag(s**-1)
        for j in range(len(t)):
            if(t[j][j] > 10):
                t[j][j] = 0
        M_inv=np.dot(v.transpose(),np.dot(t,u.transpose()))
        
        a = np.matmul(M_inv, -B)
        
        for i in range(4**qc.num_qubits):
            pauli1 = ""
            for j in range(qc.num_qubits):
                if(math.floor(i/4**j)%4 == 0):
                    pauli1 += "I"
                elif(math.floor(i/4**j)%4 == 1):
                    pauli1 += "X"
                elif(math.floor(i/4**j)%4 == 2):
                    pauli1 += "Y"
                elif(math.floor(i/4**j)%4 == 3):
                    pauli1 += "Z"
            
            if(abs(a[i]) > tolerance):
                apply_pauli(qc, pauli1, dt*a[i])

        energies.append(get_energy(qc, H))

        current_time += dt
    
    plt.plot(energies)
    plt.title("TrotterQITE")
    plt.xlabel("Timestep")
    plt.ylabel("Energy (eV)")
    plt.show()



final_time = 5.0
dt = 0.5
result = TrotterQITE(H, final_time, dt)