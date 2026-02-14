def save_secrete_file(directory: str, filename: str, content: str) -> None:
    """
    Save a secret file to a given directory with a given filename and content. The file should be accessible only by the current user.

    Args:
        directory: The directory to save the secret file.
        filename: The filename of the secret file.
        content: The content of the secret file.
    """
    # BEGIN SOLUTION
    import os

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'w') as f:
        f.write(content)
    os.chmod(os.path.join(directory, filename), 0o600)
