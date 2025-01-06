import numpy as np
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit

import math
import matplotlib.pyplot as plt

H = SparsePauliOp.from_list([ ("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5),])


def initialize_ancilla(qc, H):
    # Only work for 4 pauli terms or less
    # Would require generalizing the ansatz preparation from "Efficient Scheme for Initializing a Quantum Register with an Arbitrary Superposed State"(Long et. al. 2001) to more qubits
    a00 = float(H.coeffs[0])
    a01 = float(H.coeffs[1])
    a10 = float(H.coeffs[2])
    a11 = float(H.coeffs[3])
    
    qc.ry(np.arctan(np.sqrt((a10**2 + a11**2)/(a00**2 + a01**2))), 0)
    qc.z(0)

    qc.cry(np.arctan(a00/a01), 0, 1)
    qc.cz(0, 1)

    qc.x(0)
    qc.cry(np.arctan(a10/a11), 0, 1)
    qc.cz(0, 1)
    qc.x(0)

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
    global measurement_bit
    qc.measure([0,1], [measurement_bit, measurement_bit + 1])
    measurement_bit += 2

def measure_pauli_energy(qc, H, pauli_index):
    # Create a gate for the specific pauli string to measure energy (from Hamiltonian)
    pauli = str(H.paulis[pauli_index])
    coeff = H.coeffs[pauli_index]

    if(pauli == "I"*len(pauli)):
        return coeff

    pauli_qc = QuantumCircuit(len(pauli))
    for j in range(len(pauli)):
        if(pauli[j] == "X"):
            pauli_qc.x(j)
        if(pauli[j] == "Y"):
            pauli_qc.y(j)
        if(pauli[j] == "Z"):
            pauli_qc.z(j)
    pauli_qc = pauli_qc.control(1)
    qc.reset(0)
    qc.h(0)
    qc.compose(pauli_qc, [0, 2, 3], inplace=True)
    qc.h(0)
    qc.measure(0, 0)

    # Use AerSimulator to measure Hamiltonian Pauli String, since Sampler and Estimator primitives do not like mid-circuit measurements
    from qiskit_aer import AerSimulator
    from qiskit import transpile 

    backend = AerSimulator()
    qc_compiled = transpile(qc, backend)
    job_sim = backend.run(qc_compiled, shots=2**18)
    
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc_compiled)
    
    if("0"*(2*rounds)+"0" in counts.keys()):
        if("0"*(2*rounds)+"1" in counts.keys()):
            return coeff*(counts["0"*(2*rounds)+"0"] - counts["0"*(2*rounds)+"1"])/(counts["0"*(2*rounds)+"0"] + counts["0"*(2*rounds)+"1"])
        else:
            return -coeff
    else:
        if("0"*(2*rounds)+"1" in counts.keys()):
            return coeff
        else:
            return 0.0


def FQESS(H, rounds=1):
    # Get the gradient descent Hamiltonian
    I = SparsePauliOp.from_list([("I"*H.num_qubits, 1.0)])
    gamma = 0.15
    Hg = (I - gamma*H).simplify()

    # Normalize coefficients of Hamiltonian
    C = np.sqrt(np.sum(np.abs(Hg.coeffs)**2))
    Hg.coeffs = Hg.coeffs/C

    total_ground = 0.0
    repulsions = {}
    for pauli_index in range(len(H.coeffs)):
        global measurement_bit 
        measurement_bit = 1

        # Prepare an initial trail wavefunction
        qc = QuantumCircuit(Hg.num_qubits + math.ceil(np.log2(len(Hg.paulis))), 2*rounds + 1)
        qc.x(3)
        
        for i in range(rounds):

            # Need to first initialize the ancilla to encode the coefficients of the Hamiltonian
            initialize_ancilla(qc, Hg)
            
            # Next we need to perform controlled measurements from the ancilla to the quantum state
            measure_paulis(qc, Hg)

            # Lastly we need to perform final hadamard gates and ensure that the measurement is zero 
            final_hadamard(qc, num_ancilla=math.ceil(np.log2(len(Hg.paulis))))
            ensure_zeros(qc)

        pauli_contribution = measure_pauli_energy(qc, H, pauli_index)
        
        # Record pauli contribution to modify Hamiltonian, in order to find excited states
        repulsions[str(H.paulis[pauli_index])] = pauli_contribution
        total_ground += pauli_contribution
        

    # Now we can calculate the First Excited State
    H_repulsion = SparsePauliOp.from_list([(key, value) for key, value in repulsions.items()])
    H_excited = H - total_ground*H_repulsion
    H_excited = H_excited.simplify()

    # Get the gradient descent Hamiltonian
    I = SparsePauliOp.from_list([("I"*H_excited.num_qubits, 1.0)])
    Hg = (I - gamma*H_excited).simplify()

    # Normalize coefficients of Hamiltonian
    C = np.sqrt(np.sum(np.abs(Hg.coeffs)**2))
    Hg.coeffs = Hg.coeffs/C

    total_excited = 0.0
    for pauli_index in range(len(H.coeffs)):
        measurement_bit = 1

        # Prepare an initial trail wavefunction
        qc = QuantumCircuit(Hg.num_qubits + math.ceil(np.log2(len(Hg.paulis))), 2*rounds + 1)
        qc.x(3)

        for i in range(rounds):

            # Need to first initialize the ancilla to encode the coefficients of the Hamiltonian
            initialize_ancilla(qc, Hg)
            
            # Next we need to perform controlled measurements from the ancilla to the quantum state
            measure_paulis(qc, Hg)

            # Lastly we need to perform final hadamard gates and ensure that the measurement is zero 
            final_hadamard(qc, num_ancilla=math.ceil(np.log2(len(Hg.paulis))))
            ensure_zeros(qc)

        pauli_contribution = measure_pauli_energy(qc, H, pauli_index)
        
        # Record pauli contribution to modify Hamiltonian, in order to find excited states
        total_excited += pauli_contribution        
        
    return total_ground, total_excited

all_ground_energies = []
all_excited_energies = []
for rounds in range(5):
    ground_energy, excited_energy = FQESS(H, rounds=rounds)
    all_ground_energies.append(ground_energy)
    all_excited_energies.append(excited_energy)

    print("FQE Energy: "+str(ground_energy)+", "+str(excited_energy))

plt.plot(all_ground_energies)
plt.plot(all_excited_energies)
plt.show()