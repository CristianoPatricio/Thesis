# Zero-Shot Learning by Convex Combination of Semantic Embeddings

* É proposto um método para a construção de um sistema de *embedding* de imagens para qualquer classificador de imagens e um modelo semântico de *word embeddings* que contém as $n$ *labels* das classes no seu vocabulário;   

* As imagens são mapeadas para o espaço semântico através de combinações convexas com os *embedding vectors* das classes;

*  A chave para o ZSL é o uso de um conjunto de *embedding vectors* semânticos associados com as *labels* das classes;

* Dado um classificador standard pré-treinado, o ConSE mapeia as imagens para o espaço de *embeddings* semântico através da combinação convexa dos *embedding vectors* das *labels* das classes; Os valores das probabilidades previstas por um dado classificador para diferentes *labels* de treino são usados para calcular uma combinação pesada das *embeddings* das *labels* no espaço semântico. Isto fornece um *embedding vector* contínuo para cada imagem, que depois é usado para extrapolar as previsões feitas pelos classificadores pré-treinados, com as *labels* de treino, num conjunto de *labels* de teste.

* É também usado um modelo *skip-gram* para aprender as *embeddings* das *labels* das classes;

## ConSE: Convex combination of semantic embeddings

* Um classificador é treinado com as imagens de teste para estimar a probabilidade de uma imagem $x$ pertencer à classe $y$;
* Dado o classificador, o objetivo é transferir as previsões probabilisticas feitas pelo classificador para o conjunto das classes de teste;
* A diferença deste método para os outros é que a *embedding* prevista é baseada num classificador standard e não num modelo de regressão.
* Enquanto que o **DeViSE substitui a última camada** na rede convolucional por uma camada de transformação linear, **o ConSE mantém a camada softmax**.
  
Dada uma imagem de teste, o classificador convolucional irá devolver as $T$ top previsões do modelo (valores de probabilidade). Depois, é calculada a combinação convexa dos $T$ *embedding vectors* semânticos correspondentes e estimado o vetor semântica para a imagem de teste. De seguida, a *label* que melhor classifica a imagem é a que fica mais próxima do vetor *embedding* estimado, i.e., formalmente:

$$
\hat{y}(x,1) = \argmax_{y'\in Y_1} cos(f(x), s(y'))
$$

e $cos(f(x), s(y')) = \frac{f(x) \cdot s(y')}{\left \|  f(x) \right \| \left \|  s(y') \right \|}$

## Experiências

* Tanto o DeViSE como o ConSE fazem uso do mesmo modelo de text *skigram*, para definir o espaço das *embeddings* semânticas;
* O *skip-gram* foi treinado em 5.4 biliões de palavras da Wikipedia.org para construir *word embeddings* de 500-D;
* A CNN, a mesma que foi usada no DeViSE foi treinada no ***dataset* ImageNet 2012 1K** com 1000 classes;
* As métricas usadas para reportar os resultados foram a **flat hit@k** e a **hierarchical precision@k**;
* Foram usados 3 *datasets* de teste com dificuldade incremental ($n$-hops);
* O ConSE supera o DeViSE em todos os *datasets*;
* Acredita-se que o DeViSE tem alguma tendência para *overfitting*, uma vez que usa uma função não-linear muito complexa para mapear as imagens para o espaço semântico. Essa função está adaptada para a fase de treino, mas na fase de teste não generaliza muito bem. Em contrapartida, o ConSE usa uma estratégia simples para mapear as imagens, através de combinações convexas, que consegue generalizar melhor para classes que nunca tenham sido vistas.

#### Detalhes de implementação

* No processo de mapear os resultados da camada softmax para um *embedding vector*, o ConSE calcula primeiro a média dos *word vectors* associados a cada *label* e depois combina a média dos vetores de acordo com os valores do Softmax;

## Conclusão
* ConSE: Uma maneira determinista de incorporar imagens num espaço semântico usando as previsões probabilisticas de um classificador;
* As experiências sugerem que este modelo se comporta bem na tarefa do ZSL comparado com os algoritmos baseados na regressão.