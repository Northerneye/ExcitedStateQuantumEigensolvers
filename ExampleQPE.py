import numpy as np
from qiskit import QuantumCircuit
import qiskit
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library.standard_gates import HGate
 
estimator = Estimator()

#H = SparsePauliOp.from_list([("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5)])

qc = PhaseEstimation(4, HGate())

qc.decompose().draw(output="mpl")
plt.show()