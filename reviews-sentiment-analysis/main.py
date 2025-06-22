# from src.data_processing import DataProcessing
from src.data_processing import DataProcessor
from src.libs.utils import *

# nome_produto,UserId,nota,Time,titulo,comentario
# INPUTS
language = "en" # Idioma do texto a ser processado
file_path = "data/raw/Reviews_test.csv" # Path da base de dados
product_name_to_analyze = "B001EO5Q64" # identificador do produto a ser analisado
column_mapping = {
    "column_review_title": "titulo", # Nome da coluna do título da revisão
    "column_review_text": "comentario", # Nome da coluna do texto da revisão
    "column_review_rating": "nota", # Nome da coluna da nota da revisão
    "column_review_date": "Time", # Nome da coluna da data da revisão
    "column_product_identifier": "nome_produto" # Nome da coluna do identificador do produto

}

if __name__ == "__main__":
    
    # 1. Instanciar o objeto (rápido, não faz nada pesado)
    processor = DataProcessor(file_path=file_path, 
                              column_mapping=column_mapping, 
                              product_name=product_name_to_analyze, 
                              language=language
                              )
    # 2. Carregar os dados (lê o arquivo, pode demorar dependendo do tamanho)
    processed_dataframe = processor.process()
    print("\nDataFrame Final:")
    print(processed_dataframe)
    # print(processed_dataframe.head())
 