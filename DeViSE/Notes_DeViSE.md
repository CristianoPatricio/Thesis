# DeViSE: A Deep Visual-Semantic Embedding Model

* Neste paper é apresentado um novo modelo, o ***Deep Visual-Semantic Embedding Model***, treinado para identificar objetos visuais, usando as imagens e informação semântica recolhida de textos;

* Dada uma imagem, a classe prevista é aquela que tiver maior compatibilidade com a imagem aprensentada.
  
## 1- Introdução

* O DeViSE faz uso de dados textuais para aprender relações semânticas entre as *labels*, e mapeia as imagens para um espaço de *embeddings* semânticas.
  
## 2- Abordagem Proposta

* O método proposto contém dois conjuntos de parâmetros: (1) mapeamento linear das *features* das imagens para o espaço de *embeddings* e (2) um *embedding vector* para cada *label* possível;
* O objetivo é fazer uso do conhecimento semântico aprendido no domínio textual e transferir esse conhecimento para um modelo treinado para reconhecimento visual de objetos, i.e., associar uma imagem a uma representação semântica (*word embeddings*), através da *label* da imagem;
* Começam por pré-treinar um *neural language model* simples, adequado para aprender representações vetoriais relevantes de palavras (*word embeddings*);
* Paralelamente, é pré-treinada uma DNN para reconhecimento de objetos;
* Depois, é construído o *deep visual-semantic model*, que é uma fusão dos dois modelos pré-treinados. Dada uma nova imagem, é feita a previsão da *label* da imagem através da comparação da representação semântica da imagem com as restantes representações no espaço das *embeddings*, de forma a encontrar a representação que mais se adequa à imagem.
  
#### 2.1 Modelo de Linguagem Pré-Treinado

O modelo de linguagem aprende a representação de *embeddings* para cada termo dos dados textuais (wikipedia).

* Foi treinado um *skip-gram text model*[^1] num *corpus* de 5.7 milhões de documentos (5.4 biliões de palavras) extraídas da [wikipedia.org](https://www.wikipedia.org/);
* O texto das páginas *web* foi tokanizado num léxico de aproximadamente 155,000 termos (singulares e múltiplos) consistindo nas palavras mais comuns em Inglês e nos *datasets* usados para reconhecimento de objetos;
* O modelo *skip-gram* usa uma camada softmax hierarquica para fazer a predição de termos adjacentes e foi treinado usando um *window size* de 20 palavras;
* Foram treinados modelos *skip-gram* para várias dimensões das *embeddings*, desde 100-D a 2000-D. Foi descoberto que 500-D e 1000-D são bons valores.
  
[^1]: O skip-gram é uma das técnicas de aprendizagem não supervisionada usada para encontrar as palavras mais relacionadas para uma dada palavra. Ver https://code.google.com/archive/p/word2vec/

#### 2.2 Modelo Visual Pré-Treinado

* A arquitetura do modelo adotada é baseada no modelo vencedor do ILSVRC 2012 (ver https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). A DNN consiste em vários filtros convolucionais, camadas max-pooling, seguidas de várias camadas fully-connected treinadas com dropout. O modelo foi treinado com uma camada de output softmax, para fazer a previsão de 1 de 1000 categorias de objetos do **dataset ILSVRC 2012 1K**.
  
#### 2.3 Deep Visual-Semantic Embedding Model

Este é um modelo híbrido composto pelo modelo visual, com a camada softmax substituída por uma camada linear personalizada, que mapeia o output da última camada (4096-D) para um output de 500/1000-D, que pode ser comparado com o vetor *target* (*embedding*), e pelo modelo de linguagem.

O DeViSE é inicializado a partir dos dois modelos pré-treinados. Os *embedding vectors* aprendidos pelo modelo de linguagem são usados para mapear os termos da *label* da imagem para a representação vetorial *target*.

O modelo visual, agora sem a camada softmax, é treinado para fazer a previsão dos vetores para cada imagem, fazendo a média de uma camada de projeção e uma métrica de similaridade (cosine). A camada de projeção é uma transformação linear que mapeia uma representação de 4096-D do topo do modelo visual numa representação de 500-D ou 1000-D tal como no modelo de linguagem.

**A função de *loss*** escolhida foi uma combinação do *dot-product similarity* com a *hinge rank loss*, tal que o modelo é treinado para **produzir uma maior similaridade (*dot-product*) entre o output do modelo visual e a representação vetorial correta da *label* da imagem** do que entre o output do modelo visual e outros termos textuais *random*:

$$
loss(image,label)= \sum_{j \not ={label}} max[0,margin-\vec{t}_{label}M\vec{v}(image)+\vec{t}_jM\vec{v}(image)]
$$

onde $\vec{v}(image)$ é um vetor coluna que representa o *output* da camada de topo da rede (*core visual network*) para uma dada imagem, $M$ é a matriz dos parâmetros treinados na camada de transformação linear, $\vec{t}_{label}$ é um vetor linha que representa o *embedding vector* aprendido para o texto da *label* e $\vec{t}_j$ são os *embeddings* dos outros termos textuais. O valor do $margin$ foi definido a 0.1.

Foi usado uma *ranking loss* ao invés de uma L2 *loss* porque a avaliação do vizinho mais próximo (porblema em questão) é no fundo um problema de *ranking*.

O modelo irá calcular a similaridade fazendo o produto escalar entre o vetor devolvido pela camada de transformação e o vetor do espaço das *embeddings* que melhor representa a *label* da imagem.

O modelo DeViSE foi treinado com a descida do gradiente estocástica assíncrona numa plataforma computacional distribuída descrita em https://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf.

Na fase de teste, quando uma nova imagem é apresentada, uma parte do modelo calcula a sua representação vetorial, usando o modelo visual e a camada de transformação, e a outra parte necessita de procurar (pode ser usada uma técnica de *hashing*) as *labels* mais próximas no espaço das *embeddings*. As *labels* mais próximas são depois mapeadas de volta para os *synsets* do ImageNet para avaliação.

## 3- Resultados

* O objetivo deste trabalho é desenvolver um modelo visual que faça previsões semânticas relevantes para uma dada nova imagem;

* O DeViSE foi comparado a dois modelos com a mesma qualidade de modelo: o modelo *softmax baseline* e o modelo *random embedding*;

* Para demonstrar os resultados foram usadas as métricas: Flat Hit@K - a percentagem de imagens de teste para as quais o modelo devolve a única *label* verdadeira nas suas top $k$ previsões. Para medir a qualidade semântica das previsões em torno da *label* verdadeira, foi usada a métrica Hierarchical precision@K. 

* Na métrica Flat, o modelo *softmax baseline* atingiu maior *accuracy*, para $k=1,2$. Para $k=5,10$, o DeViSE conseguiu valores bastante idênticos ao *softmax baseline*.

* Na métrica hierárquica, o DeViSE mostrou uma melhor generalização semântica do que o *softmax baseline*, especialmente quando o $k$ é crescente.

* Uma vantagem deste modelo é a sua capacidade para inferir *labels* que nunca foram observadas, ou seja, a capacidade de generalização. Por exemplo, se o DeViSE for treinado apenas com imagens cujas *labels* são "tiger shark", "bull shark" e "blue shark", mas nunca com imagens com a *label* "shark", o modelo é capaz de generalizar para esta descrição mais ampla, uma vez que o modelo de linguagem aprendeu a representar o conceito geral de "shark" que é similar a todos os outros "sharks" específicos;
  
* Para testar esta hipótese, foram extraídas imagens do  *dataset* **ImageNet 2011 21K** que não estão presentes no *dataset* **ILSVRC 2012 1K**, onde o DeViSE foi treinado; O DeViSE obteve bom desempenho na previsão de *labels* relativas a imagens que nunca apareceram na fase de treino.

* Para quantificar a performance do modelo em dados *zero-shot*, foram construídos *datasets* de dificuldade incremental (baseada na distância *tree* do *dataset* de treino **ILSVRC 2012 1K**) a partir do *dataset* **ImageNet 2011 21K**, onde *dataset* mais fácil, "2-hop", é composto por 1589 *labels* que estão a 2 tree hops das *labels* de treino e assim sucessivamente;

* Suspeitavelmente, quanto maior for o valor $k$, ou seja, o número de top predições, mais elevado é o valor da métrica;

## Conclusão

* Foi demonstrado que o modelo proposto está apto para fazer previsões corretas de milhares de classes nunca antes vistas, através do uso de conhecimento semântico retirado de grandes fontes textuais;
* O modelo de linguagem treinado com mais dados poderá resultar num desempenho melhor do modelo;
