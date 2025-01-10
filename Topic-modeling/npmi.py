# 📊 ETAPA 1: 
# Entendendo o NPMI (Normalized Pointwise Mutual Information):
'''
O objetivo do NPMI é dizer se as palavras que constroem um tópico de fato o representa. Isto é, se as palavras presentes
no tópico representam de fato o tema do mesmo. Para isso, o NMPI é usado para medir o quanto as palavras de um tópico tendem 
a aparecer juntas nos mesmos documentos.Se as palavras de um tópico aparecem frequentemente juntas, o NMPI será mais alto, 
indicando que o tópico é mais coeso.

Resumindo: Ele mede o quão bem as palavras dentro de um tópico estão semanticamente relacionadas, 
ajudando a verificar se um tópico faz sentido.
'''

# 📋 ETAPA 2: Passo a passo prático para calcular o NPMI

# Passo 1: Importar bibliotecas necessárias
import numpy as np
from collections import Counter
from itertools import combinations

# Passo 2: Definir uma função que calcula as coocorrências das palavras em documentos

def get_word_cooccurrences(docs, topic_words):
    """
    Calcula as coocorrências de palavras em uma lista de documentos.
    
    Parâmetros:
    - docs: Lista de documentos (listas de palavras já pré-processadas)
    - topic_words: Lista de palavras do tópico a ser analisado
    
    Retorna:
    - cooc_counts: Contagem de coocorrências das palavras do tópico
    - word_counts: Contagem de aparições individuais de cada palavra
    """
    cooc_counts = Counter()
    word_counts = Counter()

    for doc in docs:
        # Filtrar palavras que pertencem ao tópico
        filtered_words = [word for word in doc if word in topic_words]
        
        # Atualizar contagem de palavras individuais
        word_counts.update(filtered_words)
        
        # Atualizar contagem de coocorrências
        for pair in combinations(filtered_words, 2):
            cooc_counts[tuple(sorted(pair))] += 1

    return cooc_counts, word_counts

# Passo 3: Definir uma função que calcula o PMI

def calculate_pmi(cooc_counts, word_counts, total_docs):
    """
    Calcula o PMI (Pointwise Mutual Information) entre pares de palavras.
    
    Parâmetros:
    - cooc_counts: Contagem de coocorrências das palavras do tópico
    - word_counts: Contagem de aparições individuais de cada palavra
    - total_docs: Número total de documentos
    
    Retorna:
    - pmi_scores: Dicionário com os pares de palavras e seus respectivos PMIs
    """
    pmi_scores = {}

    for (w1, w2), cooc in cooc_counts.items():
        p_w1 = word_counts[w1] / total_docs
        p_w2 = word_counts[w2] / total_docs
        p_w1_w2 = cooc / total_docs

        # Calcular PMI
        pmi = np.log(p_w1_w2 / (p_w1 * p_w2))
        pmi_scores[(w1, w2)] = pmi

    return pmi_scores

# Passo 4: Normalizar o PMI para obter o NMPI

def normalize_pmi(pmi_scores):
    """
    Normaliza os valores de PMI para obter o NMPI.
    
    Parâmetros:
    - pmi_scores: Dicionário com os pares de palavras e seus respectivos PMIs
    
    Retorna:
    - nmpi_scores: Dicionário com os pares de palavras e seus respectivos NMPIs
    """
    max_pmi = max(pmi_scores.values())
    min_pmi = min(pmi_scores.values())

    nmpi_scores = {pair: (pmi - min_pmi) / (max_pmi - min_pmi) for pair, pmi in pmi_scores.items()}
    
    return nmpi_scores

# 📊 ETAPA 3: Aplicar a avaliação de NMPI em tópicos gerados

# Suponha que temos os seguintes documentos pré-processados e um tópico gerado
docs = [
    ["data", "science", "machine", "learning"],
    ["machine", "learning", "model"],
    ["data", "model", "prediction"],
    ["science", "data", "learning"]
]

topic_words = ["data", "science", "machine", "learning"]

# Calcular coocorrências
cooc_counts, word_counts = get_word_cooccurrences(docs, topic_words)

# Calcular PMI
total_docs = len(docs)
pmi_scores = calculate_pmi(cooc_counts, word_counts, total_docs)

# Calcular NMPI
nmpi_scores = normalize_pmi(pmi_scores)

# Exibir resultados
print("\nCoocorrências das palavras do tópico:")
print(cooc_counts)

print("\nPMI Scores:")
print(pmi_scores)

print("\nNMPI Scores:")
print(nmpi_scores)
