import os
from typing import Union


def load_countries_names(file_path: Union[str, os.PathLike]) -> list:
    """Load countries names from file file_path into list that is returning, keeping the order.
    It is assumed that each line contains one country name."""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"Missing file {file_path}")
    except Exception as e:
        print(f"Unexpected error: {e}.")


def remove_file_if_exists(file_path: Union[str, os.PathLike]) -> None:
    """Removes file file_path if it existed."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f'Error: {e}')
