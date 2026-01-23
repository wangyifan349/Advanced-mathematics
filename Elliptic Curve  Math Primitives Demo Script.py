"""
Elliptic Curve Cryptography (ECC) Math Primitives Demo Script
=============================================================

This script provides a concise and educational implementation of ECC essentials,
suitable for mathematical learning, cryptographic prototyping, and educational use.
It implements basic elliptic curve arithmetic in Weierstrass form over a prime field:

    y^2 = x^3 + a*x + b (mod p)

Features included:

- Modular inversion, point validity check, point addition, scalar multiplication
- Secure random private key generation (decimal and hex format)
- ECC public key derivation from a private key (k*G)
- Encoding of public keys in both uncompressed (0x04 | X | Y) and compressed (0x02/0x03 | X) SEC1 formats
- Designed for secp256k1 (Bitcoin/Ethereum) and easily adaptable to other curves
- All function and variable names are formal and descriptive in English
- Inline English comments for clarity

No external dependencies except Python >=3.6 (for secrets and int.to_bytes).

How to use:

- Change parameters (a, b, p, order, G) for any curve
- Run as script for demo on secp256k1: generates random private/public key pair 
  and displays them in various formats

Author: OpenAI ChatGPT
License: MIT
"""

import secrets

INFINITY = None  # Point at infinity representation (the identity element)

def inverse_mod(k, p):
    """Compute the modular inverse of k modulo p."""
    return pow(k, -1, p)  # Python built-in modular inversion

def is_on_curve(point, a, b, p):
    """
    Check if a given point lies on the curve y^2 = x^3 + a*x + b mod p.
    :param point: tuple (x, y) or None
    :param a, b, p: curve coefficients, field prime
    :return: True if on curve or INFINITY
    """
    if point is INFINITY:
        return True  # INFINITY is always on the curve
    x, y = point
    return (y * y - (x * x * x + a * x + b)) % p == 0  # Check curve equation

def point_addition(point1, point2, a, b, p):
    """
    Add two points on the elliptic curve.
    :param point1, point2: tuples (x, y) or INFINITY
    :return: the sum as a curve point
    """
    if point1 is INFINITY:
        return point2  # O + P = P
    if point2 is INFINITY:
        return point1  # P + O = P
    x1, y1 = point1
    x2, y2 = point2
    if x1 == x2 and (y1 + y2) % p == 0:
        return INFINITY  # P + (-P) = INFINITY
    if point1 != point2:
        slope = ((y2 - y1) * inverse_mod(x2 - x1, p)) % p  # secant slope
    else:
        if y1 == 0:
            return INFINITY  # Tangent is vertical
        slope = ((3 * x1 * x1 + a) * inverse_mod(2 * y1, p)) % p  # tangent slope
    x3 = (slope * slope - x1 - x2) % p                       # Calculate new x
    y3 = (slope * (x1 - x3) - y1) % p                        # Calculate new y
    return (x3, y3)                                          # Return result

def scalar_multiplication(k, point, a, b, p):
    """
    Multiply a point by integer k via double-and-add.
    :param k: scalar multiplier
    :param point: (x, y)
    :return: k * point as (x, y)
    """
    result = INFINITY                                        # Initialize result as INFINITY
    addend = point
    while k:
        if k & 1:
            result = point_addition(result, addend, a, b, p) # Add if low bit is set
        addend = point_addition(addend, addend, a, b, p)     # Double for next bit
        k >>= 1                                              # Shift right
    return result

def generate_private_key(n):
    """
    Generate a cryptographically secure random private key, 1 <= d < n.
    :param n: order of the curve's base point
    :return: integer private key
    """
    while True:
        d = secrets.randbelow(n)                             # Generate random int in [0, n)
        if 1 <= d < n:
            return d                                         # Return if in valid range

def int_to_bytes(x, length):
    """
    Convert integer x to 'length' bytes (big endian, zero-padded).
    :param x: integer
    :param length: output bytes length
    :return: bytes
    """
    return x.to_bytes(length, 'big')                         # Big-endian representation

def encode_public_key_uncompressed(point, p):
    """
    Encode a curve point to uncompressed SEC1 format (0x04|X|Y).
    :param point: (x, y)
    :param p: field prime (determines byte length)
    :return: bytes
    """
    x, y = point
    length = (p.bit_length() + 7) // 8                       # Field size in bytes
    return b'\x04' + int_to_bytes(x, length) + int_to_bytes(y, length)     # 0x04|X|Y

def encode_public_key_compressed(point, p):
    """
    Encode a curve point to compressed SEC1 format (0x02/0x03 | X).
    :param point: (x, y)
    :param p: field prime
    :return: bytes
    """
    x, y = point
    length = (p.bit_length() + 7) // 8
    prefix = b'\x02' if y % 2 == 0 else b'\x03'              # 0x02 if even, 0x03 if odd
    return prefix + int_to_bytes(x, length)                  # prefix|X

def private_key_to_hex(priv, order):
    """
    Convert a private key integer to hex, zero-padded to byte length of curve order.
    :param priv: private key integer
    :param order: curve order (for length)
    :return: hex string
    """
    byte_len = (order.bit_length() + 7) // 8                 # bytes of order
    return f"{priv:0{byte_len * 2}x}"                        # Zero-padded hex string

def public_key_to_hex(pub_bytes):
    """
    Return the hex representation of a public key bytestring.
    :param pub_bytes: bytes
    :return: hex string prefixed 0x
    """
    return "0x" + pub_bytes.hex()                            # 0x-prefixed hex

def generate_public_key(private_key, base_point, a, b, p):
    """
    Generate the public key point from a private key scalar and generator point.
    :param private_key: integer
    :param base_point: (x, y)
    :param a, b, p: curve parameters
    :return: (x, y) public key
    """
    return scalar_multiplication(private_key, base_point, a, b, p)         # k*G

# ============== DEMONSTRATION / MAIN ===================
if __name__ == "__main__":
    # secp256k1 curve parameters (Bitcoin/Ethereum curve)
    a = 0
    b = 7
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    # Generator (Base Point) G (x, y)
    G = (
        55066263022277343669578718895168534326250603453777594175500187360389116729240,
        32670510020758816978083085130507043184471273380659243275938904335757337482424
    )
    assert is_on_curve(G, a, b, p)                            # Verify G is on curve

    # Generate cryptographically random private key
    priv_key = generate_private_key(order)
    priv_key_hex = private_key_to_hex(priv_key, order)
    print(f"Private key (int): {priv_key}")                   # Print as integer
    print(f"Private key (hex): 0x{priv_key_hex}")             # Print as hex

    # Compute corresponding public key (point)
    pub_key_point = generate_public_key(priv_key, G, a, b, p)
    assert is_on_curve(pub_key_point, a, b, p)                # Sanity check
    print("Public key coordinates (x, y):")
    print(f"  x = {pub_key_point[0]}")                        # Public key X
    print(f"  y = {pub_key_point[1]}")                        # Public key Y

    # Public key, uncompressed SEC1 format (0x04|X|Y)
    pubkey_uncompressed = encode_public_key_uncompressed(pub_key_point, p)
    print("Public key (uncompressed SEC1):", public_key_to_hex(pubkey_uncompressed)) # 0x04 format

    # Public key, compressed SEC1 format (0x02/0x03|X)
    pubkey_compressed = encode_public_key_compressed(pub_key_point, p)
    print("Public key (compressed SEC1):", public_key_to_hex(pubkey_compressed))     # 0x02/0x03 format

    # Example multiplication on curve
    k = 3
    P = scalar_multiplication(k, G, a, b, p)
    print(f"\n{k} * G (x): {P[0]}")                           # k*G X coordinate
    print(f"{k} * G (y): {P[1]}")                             # k*G Y coordinate
    print("Is k*G on curve?", is_on_curve(P, a, b, p))        # Check

    # Example addition on curve
    Q = scalar_multiplication(7, G, a, b, p)
    sum_PQ = point_addition(P, Q, a, b, p)
    print("\nP + Q:")
    print(f"x = {sum_PQ[0]}")                                 # X of sum
    print(f"y = {sum_PQ[1]}")                                 # Y of sum
    print("Is (P+Q) on curve?", is_on_curve(sum_PQ, a, b, p)) # Check on curve
