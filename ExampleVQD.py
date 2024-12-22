from qiskit_algorithms import VQD
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms.state_fidelities import ComputeUncompute

estimator = Estimator()
sampler = Sampler()
fidelity = ComputeUncompute(sampler)
optimizer = SLSQP()
H = SparsePauliOp.from_list([("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5),])

k = 2 # Number of eigenvalues to determine
betas = [10, 10] # Deflation Hyperparameter for each found state

# Define the Ansatz
ansatz = TwoLocal(2, ["rx", "ry"], "cz", reps=1)

# Run VQD and get results
vqd = VQD(estimator, fidelity, ansatz, optimizer, k=k, betas=betas)
result = vqd.compute_eigenvalues(operator=H).eigenvalues
print(result)