import json
import pennylane as qml
import pennylane.numpy as np
import scipy

def abs_dist(rho, sigma):
    """A function to compute the absolute value |rho - sigma|."""
    polar = scipy.linalg.polar(rho - sigma)
    return polar[1]

def word_dist(word):
    """A function which counts the non-identity operators in a Pauli word"""
    return sum(word[i] != "I" for i in range(len(word)))


# Produce the Pauli density for a given Pauli word and apply noise

def noisy_Pauli_density(word, lmbda):
    """
       A subcircuit which prepares a density matrix (I + P)/2**n for a given Pauli
       word P, and applies depolarizing noise to each qubit. Nothing is returned.

    Args:
            word (str): A Pauli word represented as a string with characters I,  X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.
    """


    # Put your code here #
    return "nolol"

# Compute the trace distance from a noisy Pauli density to the maximally mixed density

def maxmix_trace_dist(word, lmbda):
    """
       A function compute the trace distance between a noisy density matrix, specified
       by a Pauli word, and the maximally mixed matrix.

    Args:
            word (str): A Pauli word represented as a string with characters I, X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.

    Returns:
            float: The trace distance between two matrices encoding Pauli words.
    """


    # Put your code here #
    num_qubits = len(word)
    dev = qml.device("default.mixed", wires=num_qubits)

    op_I = qml.Identity(0)
    for i in range(1, len(word)):
        op_I = op_I @ qml.Identity(i)

    op = 0
    pauli = word[0]
    if pauli == "X":
        op = qml.PauliX(0)
    elif pauli == "Y":
        op = qml.PauliY(0)
    elif pauli == "Z":
        op = qml.PauliZ(0)
    else:
        op = qml.Identity(0)
    
    for i, pauli in enumerate(word[1:], start=1):
        if pauli == "X":
            op = op @ qml.PauliX(i)
        elif pauli == "Y":
            op = op @ qml.PauliY(i)
        elif pauli == "Z":
            op = op @ qml.PauliZ(i)
        else:
            op = op @ qml.Identity(i)

    rho_P_lambda = qml.matrix(qml.Hamiltonian(coeffs=[1/2**num_qubits, 1/2**num_qubits], observables=[op_I, op]))
    @qml.qnode(dev)
    def circuit2(dens):
        qml.QubitDensityMatrix(dens, wires=range(len(word)))
        for i in range(len(word)):
            qml.DepolarizingChannel(lmbda, wires=i)

        return qml.density_matrix(wires=range(num_qubits))
    rho_P_lambda = circuit2(rho_P_lambda)


    rho_0 = np.eye(2**num_qubits) / (2**num_qubits)

    abs_diff = abs_dist(rho_P_lambda, rho_0)
    
    # Put your code here #
    ans = np.trace(abs_diff) / 2
    return ans


def bound_verifier(word, lmbda):
    """
       A simple check function which verifies the trace distance from a noisy Pauli density
       to the maximally mixed matrix is bounded by (1 - lambda)^|P|.

    Args:
            word (str): A Pauli word represented as a string with characters I, X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.

    Returns:
            float: The difference between (1 - lambda)^|P| and T(rho_P(lambda), rho_0).
    """


    # Put your code here #
    num_pauli = word_dist(word)
    x = (1 - lmbda)**num_pauli
    
    y = maxmix_trace_dist(word, lmbda)

    return x - y


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:

    word, lmbda = json.loads(test_case_input)
    output = np.real(bound_verifier(word, lmbda))

    return str(output)


def check(solution_output: str, expected_output: str) -> None:

    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your trace distance isn't quite right!"


test_cases = [['["XXI", 0.7]', '0.0877777777777777'], ['["XXIZ", 0.1]', '0.4035185185185055'], ['["YIZ", 0.3]', '0.30999999999999284'], ['["ZZZZZZZXXX", 0.1]', '0.22914458207245006']]

for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")