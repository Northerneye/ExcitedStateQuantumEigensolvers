import qiskit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit import QuantumCircuit

import numpy as np
import math
import matplotlib.pyplot as plt

# Start by constructing the Fermionic Hamiltonian of interest
"""
molecule = MoleculeInfo(
    # Coordinates in Angstrom
    symbols=["H", "H"],
    coords=([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    multiplicity=1,  # = 2*spin + 1
    charge=0,
)
driver = PySCFDriver.from_molecule(molecule)
properties = driver.run()
transformer = FreezeCoreTransformer()#freeze_core=True, remove_orbitals=[3, 4])#remove_orbitals=[-3, -2])
#total_num_spatial_orbitals = 2
#transformer.prepare_active_space(molecule, total_num_spatial_orbitals)
problem = transformer.transform(properties)

mapper=ParityMapper(num_particles=problem.num_particles)
H = mapper.map(problem.second_q_ops()[0])

repulsion_energy = problem.nuclear_repulsion_energy + problem.hamiltonian.constants["FreezeCoreTransformer"]
input(H)
input(repulsion_energy)
""" # Here is the qubit op and repulsion energy if pyscf cannot be run (for Windows)
H = SparsePauliOp(['II', 'IZ', 'ZI', 'ZZ', 'XX'],
              coeffs=[-1.06924349+0.j,  0.26752865+0.j, -0.26752865+0.j, -0.00901493+0.j,
  0.19679058+0.j])
repulsion_energy = 0.52917721092
#"""


# Find the exact eigenvalues through diagonalization
print("Getting exact solution...")
eigenvalues = np.sort(np.linalg.eig(np.array(H.to_matrix())).eigenvalues)
exact_energies = eigenvalues[:2]



# Run VQE to find the ground state of the system |g> - return the circuit for use in later measurements
print("Running VQE to find the ground state...")
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP
from qiskit.circuit.library import TwoLocal
ansatz = TwoLocal(2, ["rx", "ry"], "cz", entanglement="linear", reps=2)
print(ansatz.decompose())

estimator = Estimator()
optimizer = COBYLA()

vqe = VQE(estimator, ansatz, optimizer)
result = vqe.compute_minimum_eigenvalue(operator=H)
vqe_ground_state_energy = result.eigenvalue + repulsion_energy
print(result)

# Get the circuit for the ground state wavefunction
vqe_qc = result.optimal_circuit.assign_parameters(result.optimal_parameters)



# Next Construct the Excitation Operators E_\mu(\alpha), which consist of all single and double excitations a^\dag*a and a^\dag*a^\dag*a*a
print("Constructing E and E^dag operators...")
from qiskit_nature.second_q.operators import FermionicOp
mapper=ParityMapper()
num_orbitals = 2

E = []
E_dag = []

# Get all single particle excitation operators
print(" -Obtaining Single Particle Excitation Operators")
for i in range(num_orbitals):
    for j in range(num_orbitals):
        E_fermionic = FermionicOp({f"+_{i} -_{j}": 1}, num_spin_orbitals=num_orbitals)
        E_qubit = mapper.map(E_fermionic)
        #print(E_qubit)
        E.append(E_qubit)
        
        # Also get the dagger operator
        E_fermionic_dag = FermionicOp({f"+_{j} -_{i}": 1}, num_spin_orbitals=num_orbitals)
        E_qubit_dag = mapper.map(E_fermionic_dag)
        E_dag.append(E_qubit_dag)

""" Get all double excitation operators
print(" -Obtaining Double Particle Excitation Operators")
for i in range(num_orbitals):
    for j in range(num_orbitals):    
        for k in range(num_orbitals):
            for l in range(num_orbitals):
                E_fermionic = FermionicOp({f"+_{i} +_{j} -_{k} -_{l}": 1}, num_spin_orbitals=num_orbitals)
                E_qubit = mapper.map(E_fermionic)
                #print(E_qubit)
                E.append(E_qubit)

                # Also get the dagger operator
                E_fermionic_dag = FermionicOp({f"+_{k} +_{l} -_{i} -_{j}": 1}, num_spin_orbitals=num_orbitals)
                E_qubit_dag = mapper.map(E_fermionic_dag)
                E_dag.append(E_qubit_dag)
#print(E)
#print(E_dag)
#"""


# Now construct the  M, Q, V, and W Matricies
print("Constructing all matricies")
# We need to construct the commutator operators [E^\dag_{\mu(\alpha)}, E_{\nu(\beta)}], and [E^\dag_{\mu(\alpha)}, H, E_{\nu(\beta)}] = 1/2*( [[E^\dag_{\mu(\alpha)}, H], E_{\nu(\beta)}] + [E^\dag_{\mu(\alpha)}, [H, E_{\nu(\beta)}]] )
# M = <g|[E^\dag_{\mu(\alpha)}, H, E_{\nu(\beta)}]|g> = 1/2*([[A, B], C] + [A, [B, C]]) 
print(" -Constructing the M matrix...")
M = np.zeros((len(E_dag), len(E)), dtype=np.complex128)
for i in range(len(E_dag)):
    for j in range(len(E)):
        op1 = 1/2 * ((((E_dag[i] & H).simplify() - (H & E_dag[i]).simplify()).simplify() & E[j]).simplify() - (E[j] & ((E_dag[i] & H).simplify() - (H & E_dag[i]).simplify()).simplify()).simplify()).simplify()
        op2 = 1/2 * ((E_dag[i] & ((H & E[j]).simplify() - (E[j] & H).simplify()).simplify()).simplify() - (((H & E[j]).simplify() - (E[j] & H).simplify()).simplify() & E_dag[i]).simplify()).simplify()
        op = (op1 + op2).simplify()
        #input(op)
        M[i, j] = estimator.run(vqe_qc, op).result().values[0]
#input(M)

# Q = -<g|[E^\dag_{\mu(\alpha)}, H, E^\dag_{\nu(\beta)}]|g>  !!!!!!!! Note the second dagger on E_\nu  !!!!!!!
print(" -Constructing the Q matrix...")
Q = np.zeros((len(E_dag), len(E_dag)), dtype=np.complex128)
for i in range(len(E_dag)):
    for j in range(len(E_dag)):
        op1 = 1/2 * ((((E_dag[i] & H).simplify() - (H & E_dag[i]).simplify()).simplify() & E_dag[j]).simplify() - (E_dag[j] & ((E_dag[i] & H).simplify() - (H & E_dag[i]).simplify()).simplify()).simplify()).simplify()
        op2 = 1/2 * ((E_dag[i] & ((H & E_dag[j]).simplify() - (E_dag[j] & H).simplify()).simplify()).simplify() - (((H & E_dag[j]).simplify() - (E_dag[j] & H).simplify()).simplify() & E_dag[i]).simplify()).simplify()
        op = (op1 + op2).simplify()
        #input(op)
        Q[i, j] = -estimator.run(vqe_qc, op).result().values[0]
#input(Q)

# V = <g|[E^\dag_{\mu(\alpha)}, E_{\nu(\beta)}]|g>
print(" -Constructing the V matrix...")
V = np.zeros((len(E_dag), len(E)), dtype=np.complex128)
for i in range(len(E_dag)):
    for j in range(len(E)):
        op = ((E_dag[i] & E[j]).simplify() - (E[j] & E_dag[i]).simplify()).simplify()
        #print(op)
        V[i, j] = estimator.run(vqe_qc, op).result().values[0]
input(V)

# W = -<g|[E^\dag_{\mu(\alpha)}, E^\dag_{\nu(\beta)}]|g>  !!!!!!!!! Note the second dagger on E_\nu  !!!!!!!
print(" -Constructing the W matrix...")
W = np.zeros((len(E_dag), len(E_dag)), dtype=np.complex128)
for i in range(len(E_dag)):
    for j in range(len(E_dag)):
        op = ((E_dag[i] & E_dag[j]).simplify() - (E_dag[j] & E_dag[i]).simplify()).simplify()
        #print(op)
        W[i, j] = -estimator.run(vqe_qc, op).result().values[0]
#input(W)



# W then need to put the matricies together -> A|psi> = E_{0n}S|psi>
print("Combining all matricies...")
A = np.zeros((2*len(E), 2*len(E)))
S = np.zeros((2*len(E), 2*len(E)))
for i in range(len(E)):
    for j in range(len(E)):
        A[i, j] = M[i, j]
        A[i+len(E), j] = Q[i, j]
        A[i, j+len(E)] = np.conjugate(Q[i, j]) # Complex Conjucate
        A[i+len(E), j+len(E)] = np.conjugate(M[i, j]) # Complex Conjucate

        S[i, j] = V[i, j]
        S[i+len(E), j] = W[i, j]
        S[i, j+len(E)] = -np.conjugate(W[i, j]) # Complex Conjucate
        S[i+len(E), j+len(E)] = -np.conjugate(V[i, j]) # Complex Conjucate


# Lastly we need to solve the generalized eigenvalue problem (like we did with QLanczos).  The lowest eigenvalue is the gap between the ground and first excited states, E_{01}.
from scipy.linalg import eigh
eigvals = eigh(A, S, eigvals_only=True)#, subset_by_value=(np.min(np.array(A)) - abs(np.min(np.array(A))/2)-1, np.min(np.array(A)) + abs(np.min(np.array(A))/2)+1))#, subset_by_index=[0, 1])
print("Lowest Energies="+str(eigvals))


# Now we have the X and Y values, and can construct the excitation operator to take the ground state to the first excited state as well! 
# (Not necessary to find E_{0n})
