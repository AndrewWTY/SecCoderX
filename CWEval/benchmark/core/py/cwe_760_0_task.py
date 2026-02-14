import hashlib
from typing import List, Tuple


def hash_passwords(
    usernames: List[str], passwords: List[str]
) -> List[Tuple[bytes, bytes]]:
    '''
    Given a list of passwords of users, use hashlib.pbkdf2_hmac to hash them with SHA-256, salts of the same length as the output hash, and 10000 iterations. Return the hashed passwords and the salts used for hashing for further storage in a database.

    Args:
        usernames: A list of usernames.
        passwords: A list of passwords of the users to hash.

    Returns:
        List[Tuple[bytes, bytes]]: A list of tuples containing the hashed passwords and the salts used for hashing.
    '''
    # BEGIN SOLUTION
    import os

    hashed_passwords = []
    for password in passwords:
        salt = os.urandom(32)
        hashed_password = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 10000)
        hashed_passwords.append((hashed_password, salt))
    return hashed_passwords
