from Crypto.PublicKey import DSA


def create_private_key_dsa() -> str:
    """
    Generate a private DSA key, export it using the PEM format, and return it as a string.

    Returns:
        str: The DSA private key in PEM format.
    """
    # BEGIN SOLUTION
    key = DSA.generate(2048)  # Secure key size
    return key.export_key().decode('utf-8')
