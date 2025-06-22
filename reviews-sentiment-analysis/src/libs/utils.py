from src.libs.constantes import *

# Identifica o tipo de arquivo e lê o conteúdo usando pandas
def read_file(file_path):

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
    
# Classificador de sentimento com base no score
def classificar_score(score: int) -> str:
    if score <= 2:
        return 'Negativo'
    elif score == 3:
        return 'Neutro'
    elif score >= 4:
        return 'Positivo'
    else:
        return 'Desconhecido'
    
# Extrator de palavras do nome do produto
def extrair_palavras_produto(nome_produto: str) -> list:
    return [
        word.lower()
        for word in nome_produto.split()
        if word not in stopwords.words('english')
        and word.isalpha()
    ]

# Função que irá tokenizar tudo em uma unica linha e remover as stopwords
def tokenizerIt(columns: str, language: str = 'en') -> str:
    stopwords = initialize_stopwords(language)  # Inicializa as stopwords com base na linguagem fornecida
    # Filtrando os tokens com base nas stopswords
    return [tokens for tokens in tokenizer.tokenize(columns) if tokens not in stopwords and tokens.isalpha()]

# Função para lematizar a lista de tokens
def lemmatize_tokens(tokens: list) -> list:
    return [lemmatizer.lemmatize(token) for token in tokens]

# inicializando o stopwords com base no input do usuário
def initialize_stopwords(language: str) -> set:
    if language == 'pt':
        return set(stopwords.words('portuguese'))
    elif language == 'en':
        return set(stopwords.words('english'))
    else:
        raise ValueError(f"Linguagem nao suportada: {language}. Linguagens suportadas são 'pt' e 'en'.")