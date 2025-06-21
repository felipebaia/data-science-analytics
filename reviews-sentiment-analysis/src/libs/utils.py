from src.libs.constantes import *


def read_file(file_path):
    """
    Reads a file and returns its contents as a pandas DataFrame, based on the file extension.

    Parameters:
        file_path (str): The path to the file to be read.

    Returns:
        pd.DataFrame: The contents of the file as a pandas DataFrame.

    Raises:
        ValueError: If the file extension is not supported.
    """
    # identifica a extensão do arquivo
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == 'csv':
        return pd.read_csv(file_path)
    elif file_extension in ['xls', 'xlsx']:
        return pd.read_excel(file_path)
    elif file_extension == 'json':
        return pd.read_json(file_path)
    elif file_extension == 'pkl':
        return pd.read_pickle(file_path)
    elif file_extension == 'parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")