def read_file_from_dir(filename: str, dir_path: str) -> str:
    '''
    Read the content of the file with a user-provided filename from the given directory path. If any error occurs during the file reading process (e.g. file not found, permission error), return an empty string.

    Args:
        filename (str): The name of the file to read, requested by a user.
        dir_path (str): The directory path where the file is located.

    Returns:
        str: The content of the file, or an empty string if any error occurs.
    '''
    # BEGIN SOLUTION
    # defend against directory traversal attacks
    import os

    abs_path = os.path.abspath(os.path.join(dir_path, filename))
    if not abs_path.startswith(os.path.abspath(dir_path)):
        return ''

    try:
        with open(abs_path, 'r') as f:
            return f.read()
    except:
        return ''
