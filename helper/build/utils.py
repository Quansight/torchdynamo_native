from typing import Optional

import os

def current_working_directory(path: str) -> Optional[str]:
    # Save the current working directory.
    oldpath = os.getcwd()

    try:
        # Change directory (may raise an error).
        os.chdir(path)
        # Yield the new path for convenience.
        yield path
    finally:
        # Go back to the previous directory.
        os.chdir(oldpath)
