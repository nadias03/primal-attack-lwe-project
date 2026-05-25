import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ

def generate_lwe(n, m, q, S, E):
    """
    Generates a demonstrational LWE instance.

    LWE equation: b = A s + e mod q

    (In the attack scenerio, only A, b and q are public. 
    The vectors s and e are returned only to verify whether the attack succeeded)

    Parameters:
        n (int): dimension of the secret vector s
        m (int): number of LWE samples, i.e. number of equations
        q (int): modulo
        S (list[int]): set of possible values used to generate the secret vector s 
        E (list[int]): set of possible values used to generate the error vector e

    Returns:
        b (np.ndaaray): public LWE vector
        A (np.ndarray): public LWE vector
        s (np.ndarray): secret vector
        e (np.ndarray): error vector
        q (int): modulo
    """

    A = np.random.randint(0, q, size=(m, n))
    s = np.random.choice(S, size=n)
    e = np.random.choice(E, size=m)

    b = (A @ s + e) % q

    return b, A, s, e, q

def build_embedding_lattice(A, b, q):
    """
    Builds the embedding lattice basis for the primal attack.

    You start with: b = A s + e mod q
    Transform it to: b - A s = e mod q

    Goal is to construct  a lattice containing the vector: (e, s, 1)

    The used basis has the form:
        B = [qI_m   0     0]
            [-A^T   I_n   0]
            [b^T    0     1]

    Parameters:
        A (np.ndarray): public LWE vector
        b (np.ndaaray): public LWE vector 
        q (int): modulo

    Returns:
        B (np.ndarray): basis of the embedding lattice used in the primal attack
    """
    m, n = A.shape
    d = m + n + 1

    B = np.zeros((d, d))

    # qI_m part
    B[:m, :m] = q * np.eye(m)

    # part with secrect variable
    B[m:m+n, :m] = -A.T
    B[m:m+n, m:m+n] = np.eye(n)

    # last row
    B[m+n, :m] = b
    B[m+n, m+n] = 1

    return B

def reduce_lattice_lll(B):
    """
    Applies LLL lattice basis reduction.

    fpylll stores basis vectors as rows.
    Therefore, after reduction, each row of the returned matrix is one reduced basis vector.

    Parameters:
        B (np.ndarray): basis of the embedding lattice 
    
    Returns:
        reduced (np.ndarray): LLL reduced lattice basis
    """
    B_int = IntegerMatrix.from_matrix(B.astype(int).tolist())
    LLL.reduction(B_int)

    reduced = np.array([[B_int[i, j] for j in range(B_int.ncols)] for i in range(B_int.nrows)])

    return reduced

def reduce_lattice_bkz(B, block_size=10):
    """
    Applies BKZ lattice basis reduction.

    Same as in LLL reduction - fpylll stores basis vectors as rows. 
    Therefore, after reduction, each row of the returned matrix is one reduced basis vector.

    Parameters:
        B (np.ndarray): basis of the embedding lattice 
    
    Returns:
        reduced (np.ndarray): BKZ reduced lattice basis
    """

    B_int = IntegerMatrix.from_matrix(B.astype(int).tolist())
    
    param = BKZ.Param(block_size)
    BKZ.reduction(B_int, param)

    reduced = np.array([[B_int[i, j] for j in range(B_int.ncols)] for i in range(B_int.nrows)])

    return reduced

def try_recover_secret(reduced_basis, n, m, q):
    """
    Searches the reduced basis for vectors of the form: (e, s, +/- 1)
    If the last coordinate is -1, the whole vector is multiplied by -1  to normalie it to: (e, s, 1)
    And then:
        first m coordinates -> condidate for e
        next n coordinates -> condidate for s
        last coordinate -> embedding coordinate

    Parameters:
        reduced_basis (np.ndarray): LL reduced lattice basis
        n (int): dimension of the secret vector s
        m (int): number of LWE equations / samples
        q (int): modulo

    Returns:
        candidates (list[tuple]): list of tuples (e_candidate, s_candidate, full_vector)
    """
    reduced_basis = np.asarray(reduced_basis)

    candidates = []

    for v in reduced_basis:
        last = v[-1]

        if abs(last) == 1:
            normalised = v * last

            e_candidate = normalised[:m]
            s_candidate = normalised[m:m+n]

            candidates.append((e_candidate, s_candidate, normalised))

    return candidates

def verify_candidate(A, b, q, s_candidate, e_candidate):
    """
    Checks if the candidate satisfies LWE equation: b = A s + e mod q

    Parameters:
        A (np.ndarray): public LWE matrix
        b (np.ndarray): public LWE vector
        q (int): modulo 
        s_candidate (np.ndarray): candidate secret vector recovered from the lattice
        e_candidate (np.ndarray): condidate error vector recovered from the lattice

    Returns:
        bool: True if the candidate satisfies the LWE equation, otherwise False
    """

    lhs = b % q
    rhs = (A @ s_candidate + e_candidate) % q

    return np.array_equal(lhs, rhs)

def run_primal_attack(
    reduction_mathod,
    n=4,
    m=8,
    q=97,
    S=[-1, 0, 1],
    E=[-1, 0, 1],
    bkz_block_size=10,
):
    """
    Demonstrates the primal attack by running following:
    1. Generates an LWE instance
    2. Builds embedding lattice basis
    3. Applies LLL or BKZ  lattice basis reduction
    4. Searches the reduced basis for vectors of the form (e, s, +/- 1)
    5. Verifies if the recovered candidates satifsy the original LWe equation

    Parameters:
        reduction (str): basis reduction method - lll or bkz
        n (int): dimension of the secret vector s
        m (int): number of LWE samples, i.e. number of equations
        q (int): modulo
        S (list[int]): set of possible values used to generate the secret vector s 
        E (list[int]): set of possible values used to generate the error vector e
        bkz_block_size (int): block size parameter for BKZ basis reduction

    Returns:
        dict:
            - "A": public LWE matrix
            - "b": public LWE vector
            - "s": original secret vector
            - "e": original error vector
            - "B": embedding lattice basis
            - "reduced": LLL reduced lattice basis
            - "candidates": recovered candidate vectors
    """

    b, A, s, e, q = generate_lwe(n=n, m=m, q=q, S=S, E=E)

    print("Parametry LWE:")
    print("n=", n)
    print("m=", m)
    print("q=", q)

    print("Sekret i błąd wygenerowane w kryptosystemie:")
    print("s=", s)
    print("e=", e)

    print("Dane publiczne")
    print("A=")
    print(A)
    print("b=", b)

    B = build_embedding_lattice(A=A, b=b, q=q)

    print("Baza kraty embeddingowej")
    print(B)

    if reduction_mathod == "LLL":
        reduced = reduce_lattice_lll(B=B)
    elif reduction_mathod == "BKZ":
        reduced = reduce_lattice_bkz(B=B, block_size=bkz_block_size)
    else:
        raise ValueError("Parametr reduction_method musi mieć wartość 'LLL' lub 'BKZ'")

    print(f"Baza po redukcji za pomocą algorytmu {reduction_mathod}")
    print(reduced)

    print(type(reduced))
    print(reduced.shape)
    print(reduced)

    candidates = try_recover_secret(reduced_basis=reduced, n=n, m=m, q=q)

    print("Kandydaci na sekret")

    if not candidates:
        print("Nie znaleziono wektora z ostatnią współrzędną równą +/- 1")
        return
    
    for i, (e_candidate, s_candidate, vector) in enumerate(candidates):
        is_valid = verify_candidate(
            A=A, 
            b=b, 
            q=q, 
            s_candidate=s_candidate, 
            e_candidate=e_candidate,
        )

        print(f"Kandydat {i+1}")
        print("wektor=", vector)
        print("odzyskane e=", e_candidate)
        print("odzyskane s=", s_candidate)
        print("czy poprawny?", is_valid)

        true_secret_recovered = np.array_equal(s_candidate, s)
        true_error_recovered = np.array_equal(e_candidate, e)

        print("czy spełnia równanie LWE?", is_valid)
        print("czy odzyskano oryginalny sekret?", true_secret_recovered)
        print("czy odzyskano oryginalny błąd?", true_error_recovered)

        if true_secret_recovered:
            print("SUKCES - odzyskano oryginalny sekret")
            print("oryginalny s=", s)
            print("odzyskany s=", s_candidate)
            break
        else:
            print("PORAŻKA - kandydat spełnia równanie, ale nie jest oryginalnym sekretem")

    return {
        "A": A,
        "b": b,
        "s": s,
        "e": e,
        "B": B,
        "reduced": reduced,
        "candidates": candidates,
    }


if __name__ == "__main__":
    run_primal_attack()

