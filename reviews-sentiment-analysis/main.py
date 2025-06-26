# from src.data_processing import DataProcessing
from src.data_processing import SentimentAnalysisModel
from src.libs.utils import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Treina ou atualiza um modelo de análise de sentimentos.")
    
    parser.add_argument(
        '--file', 
        type=str, 
        required=True, 
        help="Caminho para o arquivo CSV de entrada."
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['novo', 'atualizar'], 
        required=True, 
        help="Define o modo de operação: 'novo' para treinar do zero, 'atualizar' para alimentar um modelo existente."
    )
    parser.add_argument(
        '--language', 
        type=str, 
        choices=['en', 'pt'], 
        required=True, 
        help="Define o idioma do texto a ser processado."
    )
    # Adicione outros argumentos se necessário, como product_id
    parser.add_argument(
        '--product_id',
        type=str,
        required=False,
        help="ID do produto para gerar relatórios específicos."
    )

    args = parser.parse_args()

    # Mapeamento de colunas (exemplo)
    # Adapte este dicionário conforme o nome das colunas no seu arquivo CSV
    col_mapping = {
        "column_review_title": "titulo", # Nome da coluna do título da revisão
        "column_review_text": "comentario", # Nome da coluna do texto da revisão
        "column_review_rating": "nota", # Nome da coluna da nota da revisão
        "column_review_date": "Time", # Nome da coluna da data da revisão
        "column_product_identifier": "nome_produto" # Nome da coluna do identificador do produto
        }
 
    try:
        model_processor = SentimentAnalysisModel(
            file_path=args.file,
            column_mapping=col_mapping, # Substitua pelo seu mapeamento real
            mode=args.mode,
            product_name=args.product_id,
            language=args.language
        )

        model_processor.process()

    except (ValueError, FileNotFoundError) as e:
        print(f"\nERRO: {e}")