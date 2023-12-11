from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import SPSA, SLSQP, TNC, NFT, CG, GSLS, NELDER_MEAD
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeVigo
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms import VQE
import numpy as np
import pickle
import time

# Parameters
ansatzType = 'Two Local 1'
seed = 170
shots = [300,400,500,600,700,800,900,1000]
noise = True
optimizerTypes = ['SPSA', 'TNC', 'NM']
maxIterations = 125
trialNumber = 10

# Define Hamiltonian
H2_op = SparsePauliOp.from_list(
    [
        ("II", -1.052373245772859),
        ("IZ", 0.39793742484318045),
        ("ZI", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156),
    ]
)

# Get true ground state energy
numpy_solver = NumPyMinimumEigensolver()
result = numpy_solver.compute_minimum_eigenvalue(operator=H2_op)
trueGroundEnergy = result.eigenvalue.real

def getAnsatz(ansatzType):

    if ansatzType == 'Two Local 1':
        ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz", entanglement='linear')

    elif ansatzType == 'Two Local 2':
        ansatz = TwoLocal(rotation_blocks=["rx","rz"], entanglement_blocks="cz", entanglement='full')

    return ansatz

def getOptimizer(optimizerType, maxIterations):

    optimizerDict = {
        'SPSA': SPSA(maxiter=maxIterations),
        'SLSQP': SLSQP(maxiter=maxIterations),
        'TNC': TNC(maxiter=maxIterations),
        'NFT': NFT(maxiter=maxIterations),
        'CG': CG(maxiter=maxIterations),
        'GSLS': GSLS(maxiter=maxIterations),
        'NM': NELDER_MEAD(maxiter=maxIterations)
    }

    return optimizerDict[optimizerType]

def getEstimator (shots, seed, noise):

    if noise:

        device = FakeVigo()
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device)

        return AerEstimator(
            backend_options={
                "method": "density_matrix",
                "coupling_map": coupling_map,
                "noise_model": noise_model,
            },
            run_options={"seed": seed, "shots": shots},
            transpile_options={"seed_transpiler": seed},
        )

    else:
        return AerEstimator(
            run_options={"seed": seed, "shots": 1024},
            transpile_options={"seed_transpiler": seed},
        )

counts = []
values = []

def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)

# Run VQE
ansatz = getAnsatz(ansatzType)
vqeResults = {}
vqeTimes = {}
vqeConvergence = {}

for optimizerType in optimizerTypes:

    optimizerResults = []
    optimizerTimes = []
    optimizerConvergence = []

    for shotNumber in shots:

        shotResults = []
        shotTimes = []
        shotConvergence = []
        estimator = getEstimator(shotNumber, seed, noise)
        optimizer = getOptimizer(optimizerType, maxIterations)
        vqe = VQE(estimator, ansatz, optimizer=optimizer, callback=store_intermediate_result)

        for i in range(trialNumber):
            print(f'{optimizerType}, {shotNumber}, {i+1}')
            counts = []
            values = []
            t0 = time.time()
            shotResults.append(vqe.compute_minimum_eigenvalue(operator=H2_op).eigenvalue.real)
            t1 = time.time()
            shotTimes.append(t1-t0)
            shotConvergence.append([counts,values])

        optimizerResults.append(shotResults)
        optimizerTimes.append(shotTimes)
        optimizerConvergence.append(shotConvergence)

    vqeResults[optimizerType] = np.array(optimizerResults)
    vqeTimes[optimizerType] = np.array(optimizerTimes)
    vqeConvergence[optimizerType] = np.array(optimizerConvergence)

# Save Data
with open(f'Data/{ansatzType}_Data.pkl', 'wb') as f:
    pickle.dump([vqeResults, vqeTimes, vqeConvergence], f)
