from src.data_processing import DataProcessing
from data_processing import DataProcessor
from src.libs.utils import *

# nome_produto,UserId,nota,Time,titulo,comentario
# INPUTS
file_path = "data/raw/Reviews_test.csv"
product = "example_product"
column_mapping = {
    "column_review_title": "titulo",
    "column_review_text": "comentario",
    "column_review_rating": "nota",
    "column_review_date": "Time",
    "column_product_identifier": "nome_produto"

}

if __name__ == "__main__":
    
    # 1. Instanciar o objeto (rápido, não faz nada pesado)
    processor = DataProcessor(file_path=file_path, column_mapping=column_mapping)
    # 2. Carregar os dados (lê o arquivo, pode demorar dependendo do tamanho)
    processed_dataframe = processor.process()
    print("\nDataFrame Final:")
    print(processed_dataframe.head())
 