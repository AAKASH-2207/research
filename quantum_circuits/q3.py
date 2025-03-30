import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, IBMQ, execute
from qiskit import *
from qiskit.tools.jupyter import *
from qiskit.opflow import StateFn
from scipy.optimize import minimize
from qiskit.visualization import *
#from qiskit import BasicAer
from qiskit.quantum_info.operators import Operator

# imports from qiskit circuit
from qiskit.circuit.library import TwoLocal, RealAmplitudes, ZZFeatureMap
from qiskit.circuit import Parameter, ParameterVector

import random
import math
from qiskit_aer import Aer
# imports from qiskit nature
from qiskit_nature.drivers.second_quantization.pyscfd import PySCFDriver
from qiskit_nature.mappers.second_quantization import ParityMapper, BravyiKitaevMapper, JordanWignerMapper
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import NumPyMinimumEigensolverFactory
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit_nature.algorithms import ExcitedStatesEigensolver
from qiskit_nature.algorithms import NumPyEigensolverFactory
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
def encode(num_qubits, bl): 
    qr_m = QuantumRegister(num_qubits , 'qr_m') # for the main part - ansatz - Hc 
    cr_nn = ClassicalRegister(num_qubits, 'cr_nn') # for classical collapse and intialisation
    
    qc = QuantumCircuit(qr_m,cr_nn)
    
    qc.h(qr_m)
    qc.ry(bl, qr_m)

    return qc

def apply_ansatz(qc, qr, params,reps):

    for i in range(0,reps):
        qc.cx(qr[0], qr[1])
        qc.ry(params[4*i], qr[0])
        qc.ry(params[4*i+1], qr[1])
        qc.cx(qr[1], qr[0])
        qc.ry(params[4*i+2], qr[0])
        qc.ry(params[4*i+3], qr[1])
        qc.barrier()

def NN_atom(num_qubits,bl,parameters, backend):
    '''
    num_qubits: number of qubits for neural network
    bl: bond length 
    parameters: the initial parameters
    '''
    qr_m = QuantumRegister(num_qubits , 'qr_m') # for the main part - ansatz - Hc
    cr_nn = ClassicalRegister(num_qubits, 'cr_nn') # for classical collapse and intialisation
    
    qc = QuantumCircuit(qr_m, cr_nn)
    
    # the number of parameters should be bigger than 4 and multiples of 8
    if len(parameters)%8 != 0:
        raise ValueError('Number of parameters should be multiples of 8')
            
    # defining all the parameters, we will be using a 2 layer neural network
    params_per_layer = int(len(parameters)/2)
    params1 = ParameterVector('θ1', params_per_layer)
    params2 = ParameterVector('θ2', params_per_layer)

    
    ## encoding
    qc_1 = encode(2,bl)
    qc = qc.compose(qc_1)
    
    ## First Layer
    reps_anastz = int(len(params1)/4)
    apply_ansatz(qc, qr_m, params1,reps_anastz)

    qc.measure(qr_m, cr_nn)

    qc = qc.bind_parameters({params1 : parameters[: params_per_layer]})

    job = backend.run(qc)
    result = job.result()

    ## calculation of expectation value 
    count_dict = result.data()['counts']
    total_count = sum(count_dict.values())
    c0 = 0
    c2 = 0
    c4 = 0
    c6 = 0
    if '0x0' in count_dict.keys():
        c0 = count_dict['0x0']
    if '0x2' in count_dict.keys():
        c2 = count_dict['0x2']
    if '0x4' in count_dict.keys():
        c4 = count_dict['0x4']
    if '0x6' in count_dict.keys():
        c6 = count_dict['0x6']
    m_0_p_0 = (c0 + c4)/total_count
    m_0_p_1 = (c2 + c6)/total_count
    m_1_p_0 = (c0 + c2)/total_count
    m_1_p_1 = (c4 + c6)/total_count

    # Second layer
    qc.reset(qr_m)
    qc.h(qr_m)
    qc.ry(np.pi*(m_0_p_0 - m_0_p_1), qr_m[0])
    qc.ry(np.pi*(m_1_p_0 - m_1_p_1), qr_m[1])
    
    apply_ansatz(qc, qr_m, params2,reps_anastz)
    qc = qc.bind_parameters({params2 : parameters[params_per_layer :]})
    
    
    return qc

NN_atom(2,0.5,np.random.random(40),backend = Aer.get_backend('qasm_simulator')).draw(output='mpl')
def exact_diagonalizer(problem, converter):
    solver = NumPyMinimumEigensolverFactory()
    calc = GroundStateEigensolver(converter, solver)
    result = calc.solve(problem)
    return result

dist = np.arange(0.5,4,0.1)
exact_energy_arr = []
repulsion_energy_arr = []

for i in range(len(dist)):
    # defining of  the molecule
    molecule = "H .0 .0 .0; H .0 .0 " + str(dist[i]) 
    driver = PySCFDriver(atom=molecule)
    qmolecule = driver.run()
    
    # we define a problem using ElectronicStructureProblem
    problem = ElectronicStructureProblem(driver)
    
    # first we will calculate the repulsion energy classically 
    numpy_solver = NumPyEigensolverFactory(use_default_filter_criterion=True)
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)
    numpy_excited_states_calculation = ExcitedStatesEigensolver(converter, numpy_solver)
    numpy_results = numpy_excited_states_calculation.solve(problem)

    # our repulsion energy 
    repulsion_energy =numpy_results.nuclear_repulsion_energy
    
    
    # Generate the second-quantized operators
    second_q_ops = problem.second_q_ops()

    # Hamiltonian
    main_op = second_q_ops[0]
    mapper = ParityMapper()
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

    # The fermionic operators are mapped to qubit operators
    num_particles = problem.num_particles
    qubit_op = converter.convert(main_op, num_particles=num_particles)
    
    result_exact = exact_diagonalizer(problem, converter)
    exact_energy = np.real(result_exact.eigenenergies[0])
    exact_energy_arr.append(exact_energy + repulsion_energy )
    repulsion_energy_arr.append(repulsion_energy)
    print("Exact electronic energy", exact_energy + repulsion_energy)
plt.plot(dist, exact_energy_arr)
plt.ylabel('Energy[Hartree]')
plt.xlabel('Atomic Separation[Angstrom]')
plt.show()

dist = np.arange(0.5,4,0.1)
exact_energy = exact_energy_arr

## lets split this up into training and testing data

# training data
proportion = 1/2
training_index = np.random.randint(0,len(dist),size = int(np.ceil(len(exact_energy)*(proportion))))
training_dist = np.zeros(len(training_index))
training_energy = np.zeros(len(training_index))
for i in range(len(training_index)):
    training_dist[i] = dist[training_index[i]]
    training_energy[i] = exact_energy[training_index[i]]
    
# testing data
testing_index = np.arange(0,len(dist))
delete_index = []
for i in range(len(testing_index)):
    for j in range(len(training_dist)):
        if testing_index[i] == training_index[j]:
            delete_index.append(i)

testing_index = np.delete(testing_index,delete_index)
testing_dist = np.zeros(len(testing_index))
testing_energy = np.zeros(len(testing_index))
for i in range(len(testing_index)):
    testing_dist[i] = dist[testing_index[i]]
    testing_energy[i] = exact_energy[testing_index[i]]

def qubitOp(dist):
    
    molecule = "H .0 .0 .0; H .0 .0 " + str(dist)
    driver = PySCFDriver(atom=molecule)
    qmolecule = driver.run()

    from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
    problem = ElectronicStructureProblem(driver)

    # Generate the second-quantized operators
    second_q_ops = problem.second_q_ops()

    # Hamiltonian
    main_op = second_q_ops[0]
    mapper = ParityMapper()
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

    # The fermionic operators are mapped to qubit operators
    num_particles = problem.num_particles
    qubit_op = converter.convert(main_op, num_particles=num_particles)

    return qubit_op

def cost(qubit_op, PQC):

    qc = PQC
    backend = Aer.get_backend('statevector_simulator')
    state = execute(qc,backend).result().get_statevector()

    ## first we convert the qubit operator to matrix
    qubit_mat = qubit_op.to_matrix()

    ## basic matrix multiplication is used to obtian the expectation value.
    expectation = np.matmul(np.conjugate(state.T), np.matmul(qubit_mat , state))
    
    return expectation  ## as expectation gives us the energy we directly want to minimize this

def train_function(parameters):
    bl = training_dist
    energy = np.zeros(len(bl))   
    for i in range(len(bl)):
        qc = NN_atom(2,bl[i],parameters,backend = Aer.get_backend('qasm_simulator'))
        qubit_op = qubitOp(bl[i]) # the molecular hamiltonian for H2
        energy[i] = cost(qubit_op, qc) + repulsion_energy_arr[training_index[i]]   
    energy_net = sum(energy)/len(bl)
    return energy_net

out = minimize(train_function, x0 = np.random.random(40), method="COBYLA", options={'maxiter':2000}, tol=1e-4)
dist = np.arange(0.5,4,0.1)
exact_energy = exact_energy_arr

# parameters obtained from classical optimization procedure
parameters = out.x

# Training data
energy = np.zeros(len(training_dist))
for i in range(len(training_dist)):
    bl = training_dist
    qc = NN_atom(2,bl[i],parameters,backend = Aer.get_backend('qasm_simulator'))
    qubit_op = qubitOp(bl[i]) # molecular hamiltonian for H2
    energy[i] = cost(qubit_op, qc) + repulsion_energy_arr[training_index[i]]

## Testing data
energy_test = np.zeros(len(testing_dist))
for i in range(len(testing_dist)):
    bl_test = testing_dist
    qc = NN_atom(2,bl_test[i],parameters,backend = Aer.get_backend('qasm_simulator'))
    qubit_op_test = qubitOp(bl_test[i]) # molecular hamiltonian for H2
    energy_test[i] = cost(qubit_op_test, qc) + repulsion_energy_arr[testing_index[i]]

plt.plot(testing_dist, energy_test, '*', label = 'Testing Data')
plt.plot(training_dist, energy, 'o', label = 'Training data')
plt.plot(dist, exact_energy_arr, label = 'Exact Energy')
plt.legend()
plt.ylabel('Energy[Hartree]')
plt.xlabel('Atomic Separation[Angstrom]')
plt.show()