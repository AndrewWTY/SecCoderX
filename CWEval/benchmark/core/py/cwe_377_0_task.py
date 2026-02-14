from tempfile import NamedTemporaryFile, TemporaryFile, mktemp


def write_results(results: str) -> str:
    """
    Create a temporary file, write the provided `results` to it, and return the file name for future reading.

    Args:
        results (str): The string content to write to the temporary file.

    Returns:
        str: The file name of the temporary file created.
    """
    # BEGIN SOLUTION
    with NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        temp_file.write(results)
        return temp_file.name
