# Mantenha suas importações aqui
from src.libs.constantes import * # Cuidado com imports usando '*', é melhor importar o necessário explicitamente
from src.libs.utils import *

class DataProcessor:

    def __init__(self, file_path: str, column_mapping: Dict[str, str], product_name: Optional[str] = None, language: str = 'en') -> None:

        self.spell = Speller(lang=rf'{language}')
        self.stopwords = initialize_stopwords(language)
        self.file_path = file_path # O caminho para o arquivo a ser processado.
        self.column_mapping = column_mapping # column_mapping (Dict[str, str]): Mapeamento de colunas.
        self.productId = product_name
        self.df: Optional[pd.DataFrame] = None # O DataFrame começa como nulo
        print(f"Inicializando DataProcessor com arquivo: {self.file_path} e mapeamento de colunas: {self.column_mapping}")

    # Carrega os dados do arquivo especificado no self.file_path.
    def _load_data(self) -> None:

        print(f"Carregando dados de: {self.file_path}")
        self.df = read_file(self.file_path)

        # Verificar se o DataFrame é vazio
        if self.df is None:
            raise ValueError("O DataFrame não foi carregado corretamente. Verifique o caminho do arquivo e o formato.")
        # Verifica se o DataFrame está vazio
        if self.df.empty:
            raise ValueError("O DataFrame carregado está vazio. Verifique o arquivo de entrada.")

    # Valida se as colunas de origem esperadas existem no DataFrame. Levanta um ValueError se alguma coluna estiver faltando
    def _validate_columns(self) -> None:

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

    # Formata o texto do DataFrame, removendo espaços em branco extras e convertendo para minúsculas.
    def _format_text(self) -> pd.DataFrame:
        """
        
        """

        # Formata as colunas de texto
        self.forbbiden_words = extrair_palavras_produto(self.productId)

        if self.productId is None:
            raise ValueError("O ID do produto não foi fornecido. Por favor, especifique um ID de produto válido.")

        print("Formatando texto da review...")
        
        self.df_formated = (
            # Seleciona as colunas relevantes e aplica as transformações
            self.df.loc[self.df['ProductId'] == self.productId, ['Time', 'ProductId', 'Summary', 'Score', 'Text']]
            # reviews.loc[reviews['ProductId'] != product_id, ['Time', 'ProductId', 'Summary', 'Score', 'Text']]
            .assign(
                formated_text=lambda df: df['Text']
                    .str.replace(r'<.*?>', '', regex=True)                      # Remove HTML tags
                    .str.translate(str.maketrans('', '', string.punctuation))   # Remove prontuação
                    .str.replace(r'\s+', ' ', regex=True)                       # Remove espaços extras
                    .str.strip()                                                # Remove espaços no início e no fim
                    .str.replace(r'\b\w{1,2}\b', '', regex=True)                # Remove palavras com 1 ou 2 letras
                    .str.replace(r'\b\w{30,}\b', '', regex=True)                # Remove palavras com mais de 30 letras
                    .str.lower()                                                # Lowercase
                    .str.replace(r'\d+', '', regex=True)                        # Remove números
                    .apply(self.spell)                                               # Correção ortográfica
                    .apply(lambda text: ' '.join([w for w in text.split() if w not in self.forbbiden_words]))  # Remove palavras do nome do produto
            )
            .assign(
                classificacao=lambda df: df['Score']                            # Classifica em Pos, neu ou neg
                    .apply(classificar_score)
            )
            .assign(
                Time=lambda df: pd.to_datetime(df['Time'], unit='s')            # Convertendo timestamp para datetime
            )
            .sort_values(by='Time', ascending=False)                            # Ordena por data
            .reset_index(drop=True)                                             # Reseta o índice
        )

        # Verifica se o DataFrame formatado está vazio
        if self.df_formated.empty:
            raise ValueError("O DataFrame formatado está vazio. Verifique o ID do produto e os dados de entrada.")
        
        # Retorna o DataFrame formatado           
        return self.df_formated
    
    # Tokenizar, lemmatizar e agrupar o texto em sentimentos
    def _tokenize_and_lemmatize(self) -> None:
        self.df_tokens = (self.df_formated
 
            # Tokenizar o texto formatado para o padrão proposto
            .assign(tokens = lambda df: df['formated_text'].apply(tokenizerIt))

            # Lemmatizar os tokens (é feito antes do counter, para evitar que a msm palavra, escrita de formas diferentes, seja contabilizada)
            # Reduz palavras à sua forma base canônica (mais linguístico e inteligente) "correu -> correr"
            # O lemmatizer é mais lento, mas mais preciso que o stemmer
            .assign(lemma_text = lambda df: df['tokens'].apply(lemmatize_tokens))

            # Realizar um groupby de todas as palavras em uma unica lista
            .groupby('classificacao', as_index=False)  
            .agg({'tokens':'sum','lemma_text': 'sum'})         

            # Remoção das palavras com menos de 3 caracteres
            .assign(lemma_text = lambda df: df['lemma_text'].apply(lambda tokens: [tokens for tokens in tokens if len(tokens) > 2]))

            # Aplicando counter para saber a quantidade de palavras por classificação
            .assign(counter_lemma = lambda df: df['lemma_text'].apply(lambda tokens: Counter(tokens)))
            
            )
        
        return self.df_tokens
    
    # Gerando e salvando a nuvem de palavras
    def _generate_wordcloud(self) -> None:
        # Correção do loop e concatenação dos tokens por classificação
        for idx, row in self.df_tokens.iterrows():
            sentimento = row['classificacao']
            all_words = ' '.join(row['lemma_text'])

            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_words)

            plt.figure(figsize=(6, 3))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"{sentimento} - ProductId: '{self.productId}'", fontsize=20)
            
            # Salvar a imagem da nuvem de palavras
            filename = f"wordcloud_{sentimento}_{self.productId}.png"
            plt.savefig(os.path.join("reports", filename), bbox_inches='tight', pad_inches=0.1)
            # plt.show()

    # Gerando o gráfico de barras com a contagem de palavras por classificação
    def _generate_top_words_bar_chart(self) -> None:
        for sentimento in self.df_tokens['classificacao'].unique():    
            # 1. Juntar todas as mensagens de avaliações nota 1
            df_sentimento = self.df_formated.query(rf"classificacao == '{sentimento}'")
            texto_unico = ' '.join(df_sentimento['formated_text'])

            # 2. Processar texto com spaCy
            doc = nlp(texto_unico)

            # 3. Filtrar tokens relevantes: sem stopwords, sem pontuação, apenas palavras significativas
            palavras_filtradas_sentimento = [
                token.lemma_ for token in doc
                if token.is_alpha and                         # apenas palavras (sem números/símbolos)
                token.lemma_ not in self.stopwords and       # remove stopwords
                len(token.lemma_) > 2 and                    # remove palavras muito curtas
                token.pos_ in ['NOUN', 'VERB', 'ADJ']       # apenas substantivos, verbos e adjetivos
            ]

            # 4. Contar as 20 palavras mais frequentes
            palavras_mais_comuns = Counter(palavras_filtradas_sentimento).most_common(20)

            # 5. Criar gráfico diretamente a partir da lista de tuplas
            fig = px.bar(
                x=[item[0] for item in palavras_mais_comuns],
                y=[item[1] for item in palavras_mais_comuns],
                labels={'x': 'Palavra (com stemming)', 'y': 'Frequência'},
                title=rf'Top 20 Palavras Mais Frequentes com Stemming (Avaliações {sentimento})',
                text=[item[1] for item in palavras_mais_comuns]  # mostra valor fora da barra
            )

            # Ajustes visuais
            fig.update_traces(marker_color='indianred', textposition='outside')
            fig.update_layout(
                height=300,
                plot_bgcolor='white',
                xaxis=dict(tickfont=dict(size=14)),
                yaxis=dict(tickfont=dict(size=14))
            )

            # Salvar a imagem da nuvem de palavras
            filename = f"chart_top_words_{sentimento}_{self.productId}.png"
            fig.write_image(os.path.join("reports", filename))
            # fig.show()

    # Gera o gráfico de séries temporais com a contagem de avaliações por mês
    def _generate_tseries_line_chart(self) -> None:

        df_tseries = (self.df_formated
              .assign(Year=lambda df: df['Time'].dt.year)
              .assign(year_month=lambda df: df['Time'].dt.strftime('%Y%m'))
              .groupby(['Year', 'year_month','classificacao'], as_index=False).agg(count=('ProductId', 'count'))
              )

        df_positivo = df_tseries.query("classificacao == 'Positivo'")
        df_neutro = df_tseries.query("classificacao == 'Neutro'")
        df_negativo = df_tseries.query("classificacao == 'Negativo'")

        # Definindo o tamnho da figura
        fix, ax = plt.subplots(figsize=(10, 6))

        ax.plot(df_positivo['year_month'], df_positivo['count'], linestyle='-', color='green', label='Score')
        ax.plot(df_neutro['year_month'], df_neutro['count'], linestyle='-', color='grey', label='Score')
        ax.plot(df_negativo['year_month'], df_negativo['count'], linestyle='-', color='red', label='Score')

        # Configuracões adicionais
        ax.set_title('Gráfico de avaliação do produto', fontsize=16)
        ax.set_xlabel('Data', fontsize=14, labelpad=10)
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.set_ylabel('Score', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        # formatação do eixo X para datas
        plt.tight_layout()
        # Salvar a imagem do gráfico de séries temporais
        filename = f"tseries_line_chart_{self.productId}.png"
        plt.savefig(os.path.join("reports", filename), bbox_inches='tight', pad_inches=0.1)

    # formata os dados, removendo as linahs sem avaliacão e balancea o dataset para o conjunto de teste e treinamento
    def _format_balance_data(self) -> None:
        
        self.df['sentimento'] = self.df['Score'].apply(classificar_score) # classifica o sentimento com base na nota
        self.df = self.df.query("Text != ''")  # Remove linhas onde o texto está vazio
        df_concatenado = pd.DataFrame()

        # Balanceamento dos dados: reduz o número de avaliações caso a classe seja superior a 50% do total de avaliações
        for sentimento in self.df['sentimento'].unique():
            df = self.df.query(f"sentimento == '{sentimento}'")
            df_reduzido = pd.DataFrame()

            if len(df) > len(self.df) / 2:
                df_reduzido = df.sample(n=len(self.df) - len(df), random_state=42)  # 50% do total de avaliações serão positivas
            else:
                df_reduzido = df

            df_concatenado = pd.concat([df_reduzido, df_concatenado], ignore_index=True)

        self.df_balanceado = (df_concatenado
                    .assign(
                        formated_text=lambda df: df['Text']
                            .str.replace(r'<.*?>', '', regex=True)                      # Remove HTML tags
                            .str.translate(str.maketrans('', '', string.punctuation))   # Remove prontuação
                            .str.replace(r'\s+', ' ', regex=True)                       # Remove espaços extras
                            .str.strip()                                                # Remove espaços no início e no fim
                            .str.replace(r'\b\w{30,}\b', '', regex=True)                # Remove palavras com mais de 30 letras
                            .str.lower()                                                # Lowercase
                            .str.replace(r'\d+', '', regex=True)                        # Remove números
                            .apply(
                                lambda text: (
                                    ' '.join([w for w in text.split() if w not in self.forbbiden_words])
                                    if getattr(self, 'forbbiden_words', None)
                                    else text
                                )
                            )  # Remove palavras do nome do produto se houver
            ))
        
        # Retorna o DataFrame formatado e balanceado        
        return self.df_balanceado

    # Treinando o modelo de machine learning
    def _find_best_model(self) -> object:
        # Supondo que 'X' são suas features (texto do review) e 'y' é a coluna 'sentimento'
        X = self.df_balanceado['formated_text']
        y = self.df_balanceado['sentimento']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.25,      # Ou 0.2, 0.3, dependendo da sua preferência
            stratify=y,          # <-- O PARÂMETRO MAIS IMPORTANTE!
            random_state=42      # Para reprodutibilidade
        )

        # 1. Criar um Pipeline que combina o vetorizador e o modelo
        # Isso garante que os dados de validação dentro do GridSearchCV não "vazem" para o treinamento, tornando a avaliação mais robusta.
        pipeline = Pipeline([ 
            ('vect', CountVectorizer()),
            ('clf', MultinomialNB()),
        ])

        # 2. Definir a grade de parâmetros que você quer testar
        # A sintaxe é 'nome_da_etapa__nome_do_parâmetro'
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],  # Testa unigramas e bigramas
            'vect__max_df': (0.5, 0.75, 1.0),      # Ignora palavras muito frequentes
            'clf__alpha': (0.1, 0.5, 1.0),         # Parâmetro de suavização do Naive Bayes
        }

        # 3. Criar e treinar o GridSearchCV
        # cv=5 significa 5-fold cross-validation
        # n_jobs=-1 usa todos os processadores disponíveis para acelerar
        grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)

        # Avaliar o melhor modelo encontrado no conjunto de teste
        self.best_model = grid_search.best_estimator_
        accuracy_test = self.best_model.score(self.X_test, self.y_test)
        print("Acurácia no conjunto de teste: ", accuracy_test)

    # Gera o gráfico de matriz de confusão
    def _generate_confusion_matrix(self) -> None:
        # Predições no conjunto de teste
        self.y_pred = self.best_model.predict(self.X_test)

        # Avaliando a acurácia do modelo
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Acurácia do modelo: {accuracy:.2f}")

        # Exibindo a matriz de confusão
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.best_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.best_model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão')
        # Salvar a imagem do gráfico de séries temporais
        filename = f"confusion_matrix_chart.png"
        plt.savefig(os.path.join("reports", filename), bbox_inches='tight', pad_inches=0.1)
        
    # Salvando o modelo gerado
    def _saving_best_model(self) -> None:
        
        filename = f"modelo_sentimento_pipeline.joblib"
        joblib.dump(self.best_model, os.path.join("model", filename))

    # Orquestra o processo completo: carregar, validar e renomear
    def process(self) -> pd.DataFrame:

        self._load_data()
        self._validate_columns()
        self._rename_columns() # Retorna o self.df já renomeado
        self._format_text()
        self._tokenize_and_lemmatize()
        self._generate_wordcloud()
        self._generate_top_words_bar_chart()
        self._generate_tseries_line_chart()
        self._format_balance_data()
        self._find_best_model()
        self._saving_best_model()
        self._generate_confusion_matrix()

        print("Processamento concluído com sucesso!")
        return self.df_balanceado