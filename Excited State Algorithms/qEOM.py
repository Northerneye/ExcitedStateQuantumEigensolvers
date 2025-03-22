# Imports for PySCF and setting up H2
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo

# Imports for the Variational Algorithms
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit.circuit.library import TwoLocal
from qiskit_nature.second_q.algorithms import GroundStateEigensolver, QEOM

# Set up H2 problem using PySCF
molecule = MoleculeInfo(
    coords=([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    multiplicity=1, charge=0, symbols=["H", "H"],
)
driver = PySCFDriver.from_molecule(molecule)
properties = driver.run()
transformer = FreezeCoreTransformer()
problem = transformer.transform(properties)
mapper=ParityMapper(num_particles=problem.num_particles)

# Create the ansatz quantum circuit which will be used
ansatz = TwoLocal(problem.num_spatial_orbitals, rotation_blocks=["rx", "ry"], entanglement_blocks="cz", entanglement="linear", reps=2)

# Solve the ground state of the system with VQE
vqe = VQE(Estimator(), ansatz, COBYLA())
gse = GroundStateEigensolver(mapper, vqe)

# Run qEOM on the VQE states found
qeom_excited_states_solver = QEOM(gse, Estimator(), excitations="sd")
print(qeom_excited_states_solver.solve(problem).eigenvalues)