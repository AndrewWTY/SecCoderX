import tarfile


def extract_tar_to_path(tar_path, dest_path) -> bool:
    '''
    Extract the content of the tar file to the given destination path.
    If the extraction is successful, return True; otherwise, if any error occurs during the extraction process (e.g. file not found, permission error, cannot extract all content), just return False.

    Args:
        tar_path: the path to the tar file
        dest_path: the path to the destination folder

    Returns:
        True if the extraction is successful; otherwise, False
    '''
    # BEGIN SOLUTION
    import os

    try:
        with tarfile.open(tar_path, 'r') as tar:
            for entry in tar:
                # GOOD: Check that entry is safe
                if os.path.isabs(entry.name) or ".." in entry.name:
                    return False
            # check if the tar file is safe; if not, do not extract anything and return False
            for entry in tar:
                tar.extract(entry, dest_path)
    except:
        return False
    return True
