import numpy as np
import qiskit
import qiskit.quantum_info
from qiskit_algorithms import TimeEvolutionProblem, VarQITE
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
import scipy.linalg

var_principle = ImaginaryMcLachlanPrinciple()
estimator = qiskit.primitives.Estimator(options={"shots": 2**16})
ham = SparsePauliOp.from_list([("II", -1.5), ("IZ", 0.5), ("ZI", -0.5), ("XX", 1.5),])

# Append the tunable parameters
ansatz = qiskit.circuit.library.TwoLocal(2, ["rx", "ry"], "cz", reps=1)
params = np.random.random(ansatz.num_parameters)

# Define and Run the Time Evolution Problem
total_time = 5.0
evolution_problem = TimeEvolutionProblem(ham, total_time)#, aux_operators=[H])
qite = VarQITE(ansatz, params[:], var_principle, estimator, num_timesteps=5)
result = qite.evolve(evolution_problem)
params = result.parameter_values[1:]
#result = result.observables[-1][0][0]
#print(result)

H = np.zeros((len(params), len(params)))
S = np.zeros((len(params), len(params)))
#params_i = qiskit.circuit.ParameterVector("a", 8)
#ansatz_i = qiskit.QuantumCircuit(2)
'''
ansatz_i.rx(params_i[0], 0)
ansatz_i.rx(params_i[1], 1)
ansatz_i.ry(params_i[2], 0)
ansatz_i.ry(params_i[3], 1)
ansatz_i.cz(0,1)
ansatz_i.rx(params_i[4], 0)
ansatz_i.rx(params_i[5], 1)
ansatz_i.ry(params_i[6], 0)
ansatz_i.ry(params_i[7], 1)
'''
ansatz_i = qiskit.circuit.library.TwoLocal(2, ["rx", "ry"], "cz", reps=1)
#ansatz_i.append(ansatz_i_local, [0,1])
#ansatz_i.assign_parameters(params_i, inplace=True)
#input(qiskit.circuit.library.TwoLocal(2, ["rx", "ry"], "cz", reps=1))
#input(ansatz_i)

#params_j = qiskit.circuit.ParameterVector("b", 8)
#ansatz_j = qiskit.QuantumCircuit(2)
'''
ansatz_j.rx(params_j[0], 0)
ansatz_j.rx(params_j[1], 1)
ansatz_j.ry(params_j[2], 0)
ansatz_j.ry(params_j[3], 1)
ansatz_j.cz(0,1)
ansatz_j.rx(params_j[4], 0)
ansatz_j.rx(params_j[5], 1)
ansatz_j.ry(params_j[6], 0)
ansatz_j.ry(params_j[7], 1)
'''
ansatz_j = qiskit.circuit.library.TwoLocal(2, ["rx", "ry"], "cz", reps=1)
#ansatz_j.append(ansatz_j_local, [0,1])
#ansatz_j.assign_parameters(params_j, inplace=True)

for i in range(len(params)):
    for j in range(len(params)):
        # Compute state overlap values
        qc = qiskit.QuantumCircuit(3)
        qc.h(2)
        qc.append(ansatz_i.control(1), [2,0,1])
        qc.assign_parameters(params[i], inplace=True)
        #qc_j_dag = qiskit.QuantumCircuit(2)
        #qc_j_dag.append(ansatz, [0,1])
        #qc_j_dag.assign_parameters(params[j], inplace=True)
        qc.x(2)
        qc.append(ansatz_j.control(1), [2, 0, 1])
        qc.assign_parameters(params[j], inplace=True)
        qc.x(2)
        qc.h(2)
        #input(qc)
        
        result = estimator.run(qc, qiskit.quantum_info.SparsePauliOp.from_list([("IIZ", 1.0)])).result()
        S[i,j] = result.values[0]

qc = qiskit.QuantumCircuit(2)
qc.append(ansatz_i, [0, 1])
qc.assign_parameters(params[0], inplace=True)
result = estimator.run(qc, ham).result()
#input(result.values[0])
#input(S)
H = result.values[0]*S
#input(H)

import scipy
generalized_result = scipy.linalg.eigh(H, S)
input(generalized_result)