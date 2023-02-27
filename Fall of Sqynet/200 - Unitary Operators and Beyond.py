import json
import pennylane as qml
import pennylane.numpy as np

def W(alpha, beta):
    """ This function returns the matrix W in terms of
    the coefficients alpha and beta

    Args:
        - alpha (float): The prefactor alpha of U in the linear combination, as in the
        challenge statement.
        - beta (float): The prefactor beta of V in the linear combination, as in the
        challenge statement.
    Returns 
        -(numpy.ndarray): A 2x2 matrix representing the operator W,
        as defined in the challenge statement
    """


    # Put your code here #
    W = np.array([[np.sqrt(alpha), -np.sqrt(beta)], [np.sqrt(beta), np.sqrt(alpha)]]) / np.sqrt(alpha+beta)
    # Return the real matrix of the unitary W, in terms of the coefficients.
    return W


dev = qml.device('default.qubit', wires = 2)

@qml.qnode(dev)
def linear_combination(U, V,  alpha, beta):
    """This circuit implements the circuit that probabilistically calculates the linear combination 
    of the unitaries.

    Args:
        - U (list(list(float))): A 2x2 matrix representing the single-qubit unitary operator U.
        - V (list(list(float))): A 2x2 matrix representing the single-qubit unitary operator U.
        - alpha (float): The prefactor alpha of U in the linear combination, as above.
        - beta (float): The prefactor beta of V in the linear combination, as above.

    Returns:
        -(numpy.tensor): Probabilities of measuring the computational
        basis states on the auxiliary wire. 
    """


    # Put your code here #
    w = W(alpha, beta)

    u1 = U[0][0]
    u2 = U[0][1]
    u3 = U[1][0]
    u4 = U[1][1]
    CU  = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, u1, u2],
                [0, 0, u3, u4]])

    v1 = V[0][0]
    v2 = V[0][1]
    v3 = V[1][0]
    v4 = V[1][1]
    CV  = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, v1, v2],
                [0, 0, v3, v4]])

    qml.QubitUnitary(w, 0)
    qml.PauliX(0)
    qml.QubitUnitary(CU, wires=[0, 1])
    qml.PauliX(0)
    qml.QubitUnitary(CV, wires=[0, 1])
    qml.adjoint(qml.QubitUnitary)(w, 0)
    # Return the probabilities on the first wire
    return qml.probs(wires=0)


# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    dev = qml.device('default.qubit', wires = 2)
    ins = json.loads(test_case_input)
    output = linear_combination(*ins)[0].numpy()

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your circuit doesn't look quite right "


test_cases = [['[[[ 0.70710678,  0.70710678], [ 0.70710678, -0.70710678]],[[1, 0], [0, -1]], 1, 3]', '0.8901650422902458']]

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