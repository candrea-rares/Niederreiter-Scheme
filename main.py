import numpy as np
from itertools import combinations
import galois
import hashlib
import reedsolo
import random
from sympy import Matrix


# Helper function to generate a random binary matrix
def random_binary_matrix(rows, cols):
    return np.random.randint(0, 2, size=(rows, cols), dtype=np.uint8)


# Function to check if a square matrix is invertible (mod 2)
def is_invertible(matrix):
    det = int(np.round(np.linalg.det(matrix)))
    return det % 2 != 0


# Generate an invertible binary matrix
def generate_invertible_matrix(size):
    while True:
        matrix = random_binary_matrix(size, size)
        if is_invertible(matrix):
            return matrix


# Generate a random permutation matrix
def generate_permutation_matrix(size):
    permutation = np.random.permutation(size)
    matrix = np.eye(size, dtype=np.uint8)[permutation]
    return matrix


# Generate Goppa parity-check matrix H
def generate_goppa_parity_check_matrix(m, t, n):
    GF = galois.GF(2 ** m)  # Define the Galois Field
    g = galois.Poly.Random(t, field=GF)  # Random irreducible polynomial
    while not galois.is_irreducible(g):
        g = galois.Poly.Random(t, field=GF)

    locators = GF.Elements()[:n]  # Choose elements of GF(2^m) as locators
    np.random.shuffle(locators)
    locators = locators[:n]

    H = np.zeros((t, n), dtype=GF)  # Initialize parity-check matrix in GF(2^m)

    for i, alpha in enumerate(locators):
        beta = GF(1) / g(alpha)  # Compute reciprocal in GF(2^m)
        for j in range(t):
            H[j, i] = beta * (alpha ** j)  # Use Galois Field arithmetic

    return H.astype(np.uint8) % 2, g, locators  # Convert to binary for compatibility


# Updated Key Generation Function
def generate_keys_goppa(m, t, n):
    # Step 1: Generate Goppa code parity-check matrix H
    H, g, locators = generate_goppa_parity_check_matrix(m, t, n)

    # Step 2: Generate random invertible matrix S and permutation matrix P
    S = generate_invertible_matrix(H.shape[0])
    P = generate_permutation_matrix(H.shape[1])

    # Step 3: Compute public key H' = S * H * P
    H_prime = (S @ H @ P) % 2  # Binary arithmetic

    # Public key: H'
    # Private key: (H, S, P, g, locators)
    public_key = H_prime
    private_key = (H, S, P, g, locators)

    return public_key, private_key


# Hash function using SHA-256 and truncating to syndrome length
def hash_document(document, syndrome_length):
    hasher = hashlib.sha256()
    hasher.update(document.encode('utf-8'))
    hash_value = hasher.digest()
    hash_bits = np.unpackbits(np.frombuffer(hash_value, dtype=np.uint8))
    return hash_bits[:syndrome_length]  # Truncate to syndrome length


# Compute syndrome for given codeword and parity-check matrix
def compute_syndrome(codeword, H):
    # codeword = np.concatenate((codeword, np.zeros(t, dtype=np.uint8)))
    if codeword.ndim == 1:
        codeword = codeword[:, np.newaxis]  # Convert to column vector
    print("Codeword:", codeword.shape)
    print("H:", H.shape)
    return np.dot(H, codeword) % 2  # Matrix multiplication with codeword


# Encrypt the message using the Niederreiter cryptosystem
def encrypt_message(message, public_key, t, n):
    # Step 1: Encode the message into a binary string
    e = encode_message_no_flips(message, n, t)
    print("eT:", e)
    # Step 2: Compute the ciphertext c = Hpub * e^T (matrix multiplication)
    eT = e[:, np.newaxis]  # Transpose the message vector to be a column vector

    ciphertext = (public_key @ eT) % 2

    return ciphertext.flatten()


def create_syndrome_lookup_table(H, t):
    n = H.shape[1]
    lookup_table = {}

    # Iterate through all error patterns with weight <= t
    for weight in range(t + 1):
        for error_indices in combinations(range(n), weight):
            # Create an error pattern with the current combination of indices
            error_pattern = np.zeros(n, dtype=np.uint8)
            error_pattern[list(error_indices)] = 1

            # Compute the syndrome for the error pattern
            syndrome = (H @ error_pattern[:, np.newaxis]) % 2
            syndrome_tuple = tuple(syndrome.flatten())  # Convert to tuple for dictionary key

            # Store the error pattern in the lookup table
            lookup_table[syndrome_tuple] = error_pattern

    return lookup_table


def syndrome_decode(received_vector, private_key, lookup_table):
    # Compute the syndrome
    H, S, P, g, locators = private_key
    P_inv = P.T
    # Step 1: Compute the inverse of S (mod 2)
    S_inv = np.array(Matrix(S).inv_mod(2))

    # Step 2: Compute S^-1 * ciphertext (mod 2)
    S_inv_c = (S_inv @ ciphertext[:, np.newaxis]) % 2
    S_inv_c = S_inv_c.flatten()
    syndrome = S_inv_c
    syndrome_tuple = tuple(syndrome.flatten())  # Convert to tuple for lookup

    # Find the error pattern from the lookup table
    error_pattern = lookup_table.get(syndrome_tuple, None)
    if error_pattern is None:
        raise ValueError("Syndrome not found in lookup table. Decoding failed.")
    print("Error_pattern:", error_pattern)
    print("Received_vector:", received_vector)
    # Correct the received vector
    mT = np.dot(P_inv, error_pattern) % 2
    return mT


def decode_message(e, n):
    # Convert the binary array to a binary string
    binary_message = ''.join(map(str, e[:n]))  # Truncate to length n
    print(f"Binary message decoding: {binary_message}")  # Debug: See the full binary string

    # Group the binary string into 8-bit chunks and decode each chunk
    message = ''
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i + 8]
        if len(byte) == 8:  # Only process full bytes
            message += chr(int(byte, 2))  # Convert binary to ASCII character
            print(f"Decoded byte: {byte} -> {message[-1]}")  # Debug: See each decoded byte
    return message


def encode_message_no_flips(message, n, t):
    # Convert the message into a binary string
    binary_message = ''.join(format(ord(c), '08b') for c in message)  # Each character as 8 bits
    print(f"Binary message encoding: {binary_message}")  # Debug: See the full binary string

    # Pad or truncate the binary message to have length n
    binary_message = binary_message.ljust(n, '0')[:n]
    print(f"Padded/truncated binary message: {binary_message}")  # Debug: See the padded binary string

    # Convert the binary string into a numpy array (0 for '0' and 1 for '1')
    e = np.array([int(bit) for bit in binary_message], dtype=np.uint8)

    # Check weight constraint
    current_weight = np.sum(e)
    if current_weight > t:
        raise ValueError(f"Message weight exceeds constraint t={t}. Encoding failed.")

    return e  # Return encoded message as is


def sign_document(document, private_key, max_attempts=362880):
    H, S, P, g, locators = private_key
    n = H.shape[1]
    t = H.shape[0]

    i = 0
    while i < max_attempts:
        # Modify the document with a counter
        d_i = f"{document}|{i}"

        # Compute the hash
        hashed_document = hash_document(d_i, t)
        print("hashed_document:", hashed_document)

        # Attempt to decode the syndrome
        lookup_table = create_syndrome_lookup_table(H, t)

        try:
            decoded_codeword = syndrome_decode(hashed_document, private_key, lookup_table)
            error_raised = False  # No error occurred
        except ValueError:
            error_raised = True  # ValueError was raised

        if error_raised is not True:
            return decoded_codeword, i, hashed_document

        i += 1

    raise ValueError("Failed to find decodable syndrome within maximum attempts.")


def verify_signature(signature, public_key):
    # Extract signature components
    z, i0, hashed_document = signature

    # Compute H' * z^T (mod 2)
    z_T = z[:, np.newaxis]  # Convert z to column vector
    computed_syndrome = (public_key @ z_T).flatten() % 2  # Flatten the result

    # Check if the computed syndrome matches the hash value
    print("Signature after public encryption:", computed_syndrome)
    print("Hashed document:", hashed_document)
    return np.array_equal(computed_syndrome, hashed_document)


# Testing Example
m = 3  # GF(2^3)
t = 4  # Degree of Goppa polynomial
n = 8  # Code length
k = n - t  # Message length (approximation)
error_weight = 2

# Generate keys
public_key, private_key = generate_keys_goppa(m, t, n)
H, S, P, g, locators = private_key

print("Public Key (H'):")
print(public_key)
print("\nPrivate Key (H, S, P, g, locators):")
print("H:", H)
print("S:", S)
print("P:", P)
print("g(x):", g)
print("Locators:", locators)

# Message to encrypt
message = "Hello"

# Encrypt the message
ciphertext = encrypt_message(message, public_key, error_weight, n)
print("\nCiphertext (Encrypted Message):")
print(ciphertext)

lookup_table = create_syndrome_lookup_table(H, error_weight)
received_vector = ciphertext.copy()
# Decode the received vector
decoded_codeword = syndrome_decode(received_vector, private_key, lookup_table)
print("Decoded codeword:", decoded_codeword)
decoded_message = decode_message(decoded_codeword, n)
print("Decoded message:", decoded_message)

print("\n---- Attempt to sign -----\n")

# Document to sign
document = "Hello"

# Sign the document
signature = sign_document(document, private_key)

print("\nSignature:")
print("z:", signature[0])
print("i0:", signature[1])

print(verify_signature(signature, public_key))
