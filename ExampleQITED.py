import numpy as np
from qiskit import QuantumCircuit
import qiskit
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
import math

backend = qiskit.Aer.get_backend('qasm_simulator')
N = 2
energy_levels = 2

global shots
shots = 2**16

H = SparsePauliOp.from_list([("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5)])
alpha = [2.0 for i in range(energy_levels)] # The step size for each energy level
step_size = 0.1 # Step Size for theta_dot
max_iter = 100

def apply_param(params, parameter, qc, start=0):
    if(math.floor(parameter/2)%2 == 0):
        qc.rx(params[parameter], start + parameter%2)
    else:
        qc.ry(params[parameter], start + parameter%2)
    if(parameter == 3):
        qc.cx(start, start + 1)
    
def measure_der(parameter, qc, start = 0):
    if(math.floor(parameter/2)%2 == 0):
        qc.cx(start + N, parameter%2)
    else:
        qc.cy(start + N, parameter%2)

def pauli_measure(qc, pauli_string):
    for i in range(len(pauli_string)): # Measure Pauli Strings
        if(str(pauli_string[i]) == "X"):
            qc.cx(N,i)
        if(str(pauli_string[i]) == "Y"):
            qc.cy(N,i)
        if(str(pauli_string[i]) == "Z"):
            qc.cz(N,i)
    
def Hamiltonian_Circuit(params, pauli_string):
    qc = QuantumCircuit(N+1, 1)
    qc.h(N)
    for parameter in range(len(params)): # Apply parameterized gates
        apply_param(params, parameter, qc)
    pauli_measure(qc, pauli_string)
    qc.h(N)
    return qc

def energy(params):
    global shots
    E = 0.0
    estimator = Estimator(options={"shots": shots})
    for pauli_string in range(len(H.paulis)):
        qc = Hamiltonian_Circuit(params, H.paulis[pauli_string])
        result = estimator.run(qc, SparsePauliOp("Z"+"I"*N)).result()
        E += H.coeffs[pauli_string].real*result.values[0]
    return E

def A_Circuit(params, i, j):
    qc = QuantumCircuit(N+1, 1)
    qc.h(N)
    for parameter in range(len(params)):# Apply parameterized gates
        if(parameter == i):
            qc.x(N)
            measure_der(parameter, qc) # Measure generators
            qc.x(N)
        if(parameter == j):
            measure_der(parameter, qc) # Measure second generators
        apply_param(params, parameter, qc)
    qc.h(N)
    return qc

def Measure_A(params):
    global shots
    observable = SparsePauliOp("Z"+"I"*N)
    estimator = Estimator(options={"shots": shots})
    A = [[0.0 for i in range(len(params))] for j in range(len(params))]
    for i in range(len(params)):
        for j in range(len(params)-i):
            qc = A_Circuit(params, i, i+j)
            result = estimator.run(qc, observable).result()
            A[i][i+j] = 1/4*result.values[0]
            A[i+j][i] = 1/4*result.values[0]
    return A

def C_Circuit(params, i, pauli_string):
    qc = QuantumCircuit(N+1, 1)
    qc.h(N)
    qc.s(N)#To get only imaginary component
    for parameter in range(len(params)): # Apply parameterized gates
        if(parameter == i):
            qc.x(N)
            measure_der(parameter, qc) # Measure generators
            qc.x(N)
        apply_param(params, parameter, qc)
    pauli_measure(qc, pauli_string)
    qc.h(N)
    return qc

def Measure_C(params):
    global shots
    C = [0.0 for i in range(len(params))]
    observable = SparsePauliOp("Z"+"I"*N)
    repulsion_observable = SparsePauliOp("Z"+"I"*(2*N))
    estimator = Estimator(options={"shots": shots})
    for i in range(len(params)):
        for pauli_string in range(len(H.paulis)):
            qc = C_Circuit(params, i, H.paulis[pauli_string])
            result = estimator.run(qc, observable).result()
            C[i] -= 1/2*H.coeffs[pauli_string].real*result.values[0]
        for level in range(len(all_params)):
            repulsionqc = Deflation_Circuit(params, i, all_params[level])
            result = estimator.run(repulsionqc, repulsion_observable).result()
            C[i] -= 1/2*alpha[level]*result.values[0]
    return C

def Deflation_Circuit(params, i, all_param):
    qc = QuantumCircuit(2*N+1, 1)
    qc.h(2*N)
    qc.s(2*N) # To get only imaginary component
    for parameter in range(len(params)): # Apply parameterized gates
        if(parameter == i):
            qc.x(2*N)
            measure_der(parameter, qc, start=N)
            qc.x(2*N)
        apply_param(params, parameter, qc)
    # Prepare previous energy level
    for parameter in range(len(all_param)):
        apply_param(all_param, parameter, qc, start=N)
    # Controlled Swap Test
    for qubit in range(N):
        qc.cswap(2*N, qubit, N+qubit)
    qc.h(2*N)
    return qc

energies = []
all_energies = [[] for i in range(energy_levels)]
all_params = []
for energy_level in range(energy_levels): # Add the theta_dots for each energy level
    my_params = (2*np.pi*np.random.rand(8)).tolist() # Reset Initial Parameters after each run
    for i in range(max_iter):
        theta_dot = np.array([0.0 for j in range(len(my_params))])
        cascade = 0
        A = np.array(Measure_A(my_params))
        C = np.array(Measure_C(my_params))

        #Approximately invert A using Truncated SVD
        u,s,v=np.linalg.svd(A)
        for j in range(len(s)): 
            if(s[j] < 0.01):
                s[j] = 1e6
        t = np.diag(s**-1)
        A_inv=np.dot(v.transpose(),np.dot(t,u.transpose()))
        theta_dot = np.matmul(A_inv, C)
        
        print()
        print("Energy Level = "+str(energy_level))
        print("Iteration = "+str(i))
        print("Energy: "+str(energy(my_params)))

        for j in range(len(theta_dot)):
            my_params[j] += theta_dot[j]*step_size
        all_energies[energy_level].append(energy(my_params))
    energies.append(energy(my_params))
    all_params.append(my_params[:])
print("QITED Energies: "+str(energies))

for energy_level in range(energy_levels):
    plt.plot([i for i in range(len(all_energies[energy_level]))], all_energies[energy_level])
plt.title("QITED")
plt.xlabel("Timestep")
plt.ylabel("Energy")
plt.show()