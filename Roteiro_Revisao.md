---
marp: true
---

Roteiro de Revisão - Projeto NLP
Construindo um Pipeline de Recuperação de Informação com Re-ranking


Índice

    Introdução

    Fundamentos de NLP e Processamento de Documentos

    Extração de Texto de PDFs

    Gold Standard – Criando Dados de Avaliação

    Embeddings e Busca Semântica

    LangChain na Prática

    Two-Stage Retrieval – Retrieve + Re-rank

    Avaliação de Pipeline RAG

    Resumo e Próximos Passos

    Referências

1. Introdução <a name="introdução"></a>

Este ebook reúne os principais conceitos e práticas desenvolvidos ao longo do projeto de NLP (Processamento de Linguagem Natural) do grupo. O objetivo é consolidar o aprendizado sobre como construir um sistema de Recuperação Aumentada por Geração (RAG) que extrai informações de documentos PDF, realiza busca semântica e utiliza re‑ranking para obter respostas precisas.

O material inclui desde a extração de texto de PDFs com PyMuPDF até a criação de um pipeline completo com LangChain, Sentence Transformers e Ollama. Todos os exemplos de código são baseados nas atividades registradas no diário de bordo do projeto.
2. Fundamentos de NLP e Processamento de Documentos <a name="fundamentos"></a>
2.1 O que é NLP?

NLP (Natural Language Processing) é a área da Inteligência Artificial que capacita computadores a entender, interpretar e gerar linguagem humana. No contexto do projeto, usamos NLP para:

    Extrair texto de documentos não estruturados (PDFs).

    Transformar o texto em representações numéricas (embeddings).

    Responder perguntas com base no conteúdo dos documentos.

2.2 Arquitetura RAG (Retrieval-Augmented Generation)

Um sistema RAG combina duas etapas principais:

    Recuperação (Retrieval): Busca trechos relevantes em uma base de documentos.

    Geração (Generation): Utiliza um modelo de linguagem (LLM) para produzir uma resposta final baseada nos trechos recuperados.

2.3 Chunking

Documentos grandes são divididos em pedaços menores chamados chunks. Isso é necessário porque:

    Modelos de embedding têm limite de tamanho de entrada.

    A busca é mais eficiente em trechos curtos.

    Permite identificar com precisão qual parte do documento é relevante.

3. Extração de Texto de PDFs <a name="extração"></a>
3.1 Ferramenta PyMuPDF (fitz)

A biblioteca pymupdf permite extrair texto, imagens e metadados de PDFs de forma simples e eficiente.
python

import pymupdf

def extrair_texto_pdf(caminho_pdf):
    doc = pymupdf.open(caminho_pdf)
    texto_completo = ""
    for pagina_num in range(len(doc)):
        pagina = doc[pagina_num]
        texto = pagina.get_text()
        texto_completo += f"\n--- Página {pagina_num+1} ---\n"
        texto_completo += texto
    return texto_completo

# Uso
texto = extrair_texto_pdf("contrato.pdf")
print(f"Total de páginas: {texto.count('--- Página')}")

3.2 Limpeza de Texto

Após a extração, é comum aplicar técnicas de limpeza:

    Remover cabeçalhos/rodapés repetitivos.

    Normalizar quebras de linha e espaços.

    Substituir caracteres especiais.

4. Gold Standard – Criando Dados de Avaliação <a name="goldstandard"></a>
4.1 O que é um Gold Standard?

É um conjunto de pares pergunta–resposta que serve como referência para avaliar o sistema. Idealmente, cada resposta deve estar associada à posição exata (início e fim) no documento original, permitindo métricas precisas.
4.2 Exemplo de Estrutura
python

qa_pairs = [
    {
        "question": "Quais são as vantagens do endereçamento multicast?",
        "answer": "O multicast transmite um único pacote para todos os nós...",
        "answer_start": 1234,
        "answer_end": 5678
    },
    {
        "question": "Como funciona o campo Hop Limit no IPv6?",
        "answer": "O Hop Limit evita loops sendo decrementado a cada roteador...",
        "answer_start": 8901,
        "answer_end": 10234
    }
]

4.3 Importância das Posições

Registrar as posições permite avaliar se o retrieval encontrou exatamente o trecho correto (IoU = 1), não apenas um texto semanticamente similar.
5. Embeddings e Busca Semântica <a name="embeddings"></a>
5.1 O que são Embeddings?

Embeddings são vetores de números reais que representam o significado de um texto. Textos com significados próximos ficam próximos no espaço vetorial.
5.2 Modelos de Sentence Transformers

Testamos três modelos principais:
Modelo	Uso	Métrica de Similaridade
multi-qa-mpnet-base-cos-v1	Busca semântica	Cosseno
multi-qa-mpnet-base-dot-v1	Busca semântica	Produto escalar
msmarco-bert-base-dot-v5	Passagens MSMARCO	Produto escalar
5.3 Exemplo Prático
python

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

docs = [
    "O Hop Limit evita pacotes em loop na rede",
    "O multicast economiza banda transmitindo um único pacote"
]
doc_embs = model.encode(docs)
query_emb = model.encode("Como evitar pacotes perdidos?")

scores = util.cos_sim(query_emb, doc_embs)[0]
print(scores)  # [0.85, 0.12]

6. LangChain na Prática <a name="langchain"></a>
6.1 O Ecossistema LangChain

LangChain oferece componentes modulares para construir pipelines de NLP. Usamos:

    Document Loaders: Carregar PDFs com PyPDFLoader.

    Text Splitters: Dividir documentos em chunks (ex: RecursiveCharacterTextSplitter).

    Embeddings: Interface com modelos Hugging Face (HuggingFaceEmbeddings).

    Vector Stores: Armazenar embeddings (Chroma, FAISS).

6.2 Pipeline Completo
python

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Carregar
loader = PyPDFLoader("documento.pdf")
docs = loader.load()

# Chunkar
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True   # guarda posição no documento original
)
chunks = text_splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Armazenar
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Buscar
resultados = vector_store.similarity_search(
    "O que é multicast?",
    k=3
)

6.3 Adicionando Metadados

É útil preservar metadados como posição, página e nome do documento para avaliação.
7. Two-Stage Retrieval – Retrieve + Re-rank <a name="twostage"></a>
7.1 Motivação

    Bi-Encoder (retrieve) é rápido, mas menos preciso.

    Cross-Encoder (re-rank) é mais preciso, mas lento para grandes bases.
    A combinação oferece velocidade + precisão.

7.2 Como Funciona

    Retrieve: Bi-Encoder recupera top‑k candidatos (ex: 10) rapidamente.

    Re-rank: Cross-Encoder reordena esses candidatos, selecionando os melhores (ex: 3).

7.3 Implementação com Sentence Transformers
python

from sentence_transformers import SentenceTransformer, CrossEncoder, util

bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def two_stage_search(query, corpus, top_k_retrieve=10, top_k_rerank=3):
    # Retrieve
    query_emb = bi_encoder.encode(query, convert_to_tensor=True)
    corpus_embs = bi_encoder.encode(corpus, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, corpus_embs, top_k=top_k_retrieve)[0]
    
    # Re-rank
    candidates = [corpus[hit["corpus_id"]] for hit in hits]
    pairs = [[query, cand] for cand in candidates]
    scores = cross_encoder.predict(pairs)
    
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k_rerank]

7.4 Integração com LangChain

Para usar com LangChain, podemos empacotar o CrossEncoderReranker (da langchain_classic) ou criar um retriever personalizado.
8. Avaliação de Pipeline RAG <a name="avaliação"></a>
8.1 Métricas Fundamentais
Métrica	Descrição	Fórmula / Cálculo
Precision@K	Fração de resultados relevantes entre os K recuperados.	acertos / K
Recall@K	Fração de documentos relevantes recuperados entre todos os relevantes.	acertos / total_relevantes
IoU (Interseção sobre União)	Similaridade textual entre resposta gerada e gold.	|palavras_resposta ∩ palavras_gold| / |palavras_resposta ∪ palavras_gold|
Similaridade Semântica	Cosseno entre embeddings da resposta gerada e gold.	cos_sim(emb_resp, emb_gold)
8.2 Avaliação com Posições
python

def calcular_overlap(chunk_start, chunk_end, ans_start, ans_end):
    inter_inicio = max(chunk_start, ans_start)
    inter_fim = min(chunk_end, ans_end)
    inter = max(0, inter_fim - inter_inicio)
    return inter / (ans_end - ans_start) if ans_end > ans_start else 0

def avaliar_retrieval(pergunta, gold_start, gold_end, chunks_retornados):
    for chunk in chunks_retornados:
        overlap = calcular_overlap(
            chunk.metadata["start"],
            chunk.metadata["end"],
            gold_start,
            gold_end
        )
        if overlap == 1.0:
            return True  # chunk exato recuperado
    return False

8.3 Avaliação Completa do Pipeline
python

def avaliar_pipeline(qa_pairs, retriever, llm):
    metricas = {"retrieval": [], "semantic": [], "iou": []}
    for qa in qa_pairs:
        # Retrieve
        chunks = retriever.invoke(qa["question"])
        context = " ".join([c.page_content for c in chunks])
        # Generate
        resposta = llm.generate(qa["question"], context)
        # Métricas
        retrieval_score = calcular_similaridade(qa["question"], context)
        semantic_sim = cosine_similarity(encode(resposta), encode(qa["answer"]))
        iou = calcular_iou(resposta, qa["answer"])
        
        metricas["retrieval"].append(retrieval_score)
        metricas["semantic"].append(semantic_sim)
        metricas["iou"].append(iou)
    
    return {k: np.mean(v) for k, v in metricas.items()}

9. Resumo e Próximos Passos <a name="resumo"></a>
9.1 Arquitetura Final
text

PDF → Extração → Chunking → Embeddings → Vector Store → Retrieve → Re-rank → Geração
 ↑        ↑          ↑            ↑            ↑           ↑          ↑         ↑
PyMuPDF  Limpeza  TextSplitter  Bi-Encoder  Chroma/FAISS  Rápido  Cross-Encoder  LLM

9.2 Principais Aprendizados

    Chunking influencia diretamente a qualidade do retrieval.

    Two-stage retrieval equilibra velocidade e precisão.

    Avaliação com posições é mais rigorosa que similaridade semântica.

    LLMs locais (via Ollama) permitem testes sem dependência de APIs.

9.3 Ferramentas Utilizadas

    PyMuPDF – extração de texto.

    Sentence Transformers – embeddings e busca.

    LangChain – orquestração do pipeline.

    Chroma / FAISS – vector stores.

    Ollama – modelos LLM locais.

    Cross-Encoder – re‑ranking.

9.4 Próximos Passos Sugeridos

    Expandir o gold standard para mais de 100 perguntas.

    Implementar avaliação com posições usando DocAnno.

    Testar diferentes estratégias de chunking (tamanho, overlap, por parágrafo).

    Experimentar modelos maiores, como PORTULAN/albertina-1b5-portuguese-ptbr-encoder.

    Criar um dashboard para visualizar métricas e comparar versões do pipeline.

10. Referências <a name="referências"></a>

    PyMuPDF Documentation

    Sentence Transformers

    LangChain Docs

    Ollama

    Chroma

    Cross-Encoder Models

Apêndice: Exemplo Completo de Pipeline com Avaliação
python

# ================================
# Instalação (em notebook)
# ================================
# !pip install -qU langchain sentence-transformers faiss-cpu langchain_text_splitters langchain_community ollama

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ollama

# ================================
# 1. Gold Standard (exemplo resumido)
# ================================
qa_pairs = [
    {
        "question": "Quais são as vantagens do multicast?",
        "answer": "O multicast transmite um único pacote para todos os nós, economizando banda.",
        "answer_start": 0,
        "answer_end": 50
    },
    # mais pares...
]

# ================================
# 2. Embeddings e Indexação
# ================================
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [qa["answer"] for qa in qa_pairs]
doc_embs = model.encode(documents)

def retrieve_context(question, top_k=3):
    q_emb = model.encode([question])
    scores = cosine_similarity(q_emb, doc_embs)[0]
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [(documents[i], scores[i]) for i in top_idx]

# ================================
# 3. Geração com LLM local
# ================================
def generate_answer(question, context):
    prompt = f"Pergunta: {question}\nContexto: {context}\nResposta:"
    response = ollama.generate(model='llama3.2:1b', prompt=prompt)
    return response['response']

# ================================
# 4. Avaliação
# ================================
results = []
for qa in qa_pairs:
    context, score = retrieve_context(qa["question"])[0]
    answer = generate_answer(qa["question"], context)
    sim = cosine_similarity(model.encode([answer]), model.encode([qa["answer"]]))[0][0]
    results.append({"question": qa["question"], "score": score, "similarity": sim})

# ================================
# 5. Métricas finais
# ================================
avg_score = np.mean([r["score"] for r in results])
avg_sim = np.mean([r["similarity"] for r in results])
print(f"Avg Retrieval Score: {avg_score:.3f}")
print(f"Avg Semantic Similarity: {avg_sim:.3f}")

Fim do Ebook
Como gerar o PDF

    Salve este conteúdo em um arquivo chamado ebook_revisao_nlp.md.

    Converta para PDF usando uma das opções abaixo:

        Pandoc (linha de comando):
        bash

        pandoc ebook_revisao_nlp.md -o ebook_revisao_nlp.pdf --pdf-engine=xelatex

        Markdown to PDF (extensão do VS Code ou plugin).