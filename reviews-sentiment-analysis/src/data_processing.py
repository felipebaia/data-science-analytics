# Mantenha suas importações aqui
from src.libs.constantes import * # Cuidado com imports usando '*', é melhor importar o necessário explicitamente
from src.libs.utils import read_file

class DataProcessor:

    def __init__(self, file_path: str, column_mapping: Dict[str, str]):

        self.file_path = file_path # O caminho para o arquivo a ser processado.
        self.column_mapping = column_mapping # column_mapping (Dict[str, str]): Mapeamento de colunas.
        self.df: Optional[pd.DataFrame] = None # O DataFrame começa como nulo

    # Carrega os dados do arquivo especificado no self.file_path.
    def _load_data(self) -> None:

        print(f"Carregando dados de: {self.file_path}")
        self.df = read_file(self.file_path)

    # Valida se as colunas de origem esperadas existem no DataFrame. Levanta um ValueError se alguma coluna estiver faltando
    def _validate_columns(self) -> None:

        if self.df is None:
            raise ValueError("O DataFrame não foi carregado. Chame o método de carga primeiro.")

        expected_source_columns = set(self.column_mapping.values())
        actual_columns = set(self.df.columns)
        
        missing_columns = expected_source_columns - actual_columns
        if missing_columns:
            raise ValueError(f"As seguintes colunas esperadas não foram encontradas no arquivo: {missing_columns}")

    # Renomeia as colunas do DataFrame com base no mapeamento fornecido
    def _rename_columns(self) -> None:
        
        if self.df is None:
            raise ValueError("O DataFrame não foi carregado.")
        
        # Mapping para renomear as colunas
        mapping = dict(zip(self.column_mapping.values(), column_name_mapping.values()))
        
        print("Renomeando colunas...")
        # A função rename do pandas já espera um dicionário {antigo: novo}
        self.df = self.df.rename(columns=mapping)

    # Orquestra o processo completo: carregar, validar e renomear
    def process(self) -> pd.DataFrame:

        self._load_data()
        self._validate_columns()
        self._rename_columns() # Retorna o self.df já renomeado
        
        print("Processamento concluído com sucesso!")
        return self.df