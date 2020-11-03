# Latent Embeddings for Zero-Shot Classification (CVPR 2016)

## Key ideas

* Método que aprende uma **função de compatibilidade entre uma imagem** e a respetiva **classe**;
* O modelo foi treinado com uma **função objetivo baseada em *ranking***;
* As classes de treino e de teste estão interligadas através de alguma informação auxiliar, p.e., atributos;
* A anotação de imagens é um processo custoso, uma vez que requer opinião especializada e um grande número de atributos. Para superar esta limitação, os trabalhos recentes têm explorado representações textuais que são aprendidas a partir de um *text corpus*, p.e., Wikipedia;
* O progresso feito na classificação de imagens no domínio do ZSL deve-se, sobretudo, às *features* das imagens baseadas em *Deep Learning* e à aprendizagem de funções de compatibilidade discriminativas entre as imagens e as classes;
* O objetivo deste método (LatEm) é melhorar a *framework* de aprendizagem de compatibilidade, usando informação auxiliar não supervisionada (*word vectors*);
* A ideia principal das *embedding frameworks* é primeiramente representar as imagens e as classes em espaços de vetores multidimensionais:
  * As ***embeddings* das imagens** são obtidas a partir de **CNNs *State-of-the-Art***;
  * As ***embeddings* das classes** podem ser obtidas usando **atributos** ou extraídas a partir de um ***text corpus***.

## Funções de Compatibilidade

* A **função bilinear de compatibilidade** é aprendida tal que as imagens pertencentes à mesma classe fiquem todas agrupadas e as imagens de classes diferentes estejam "longe" umas das outras. Depois de **aprendida**, pode ser **usada para prever a classe de uma dada imagem de teste**;
* No entanto, uma função de compatibilidade linear não é adequada para endereçar o problema da classificação *fine-grained*. Neste caso, é necessário um modelo que consiga agrupar os objetos com propriedades similares e, para cada grupo de objetos, aprender um modelo de compatibilidade. O método proposto incorpora *latent variables* para aprender uma função de compatibilidade linear por partes entre uma imagem e a classe respetiva;
* É usado um SGD eficiente para a aprendizagem do modelo;
* :warning: Os métodos baseados em classificadores de atributos são considerados sub-ótimos, uma vez que a sua fiabilidade no mapeamento binário entre os atributos e imagens causa perda de informação;

## *Bilinear Joint Embeddings*

* Dado um conjunto de treino $T$, onde $x$ é a *embedding* da imagem e $y$ a *embedding* da classe, a função de previsão que escolhe a classe com maior compatibilidade é:

$$
f(x) = \argmax_{y \in Y}F(x,y)
$$

com $F(x,y) = x^TWy$, onde a matriz $W \in \Reals^{d_x \times d_y}$ é o parâmetro a ser aprendido a partir dos dados de treino.

## Latent Embeddings Model (LatEm)

$$
F(x,y) = \max_{1 \le i \le K}x^TW_iy
$$

O objetivo é aprender $i$ matrizes $W$ tal que para um dado par (imagem, classe) seja atingido o máximo *score* com a matriz $i$, uma vez que foi provado que diferentes $W_i$ capturam caraterísticas visuais distintas dos objetos, i.e., a cor, a forma, etc.

* O objetivo é aprender um conjunto de espaços de compatibilidade que minimizam o risco:

$$
\frac{1}{N} \sum_{n=1}^{\left| T \right|} L(x_n,y_n)
$$

$$
L(x_n,y_n) = \sum_{y\in Y} max\{0, \Delta(y_n,y)+F(x_n,y)-F(x_n,y_n)\}
$$

* O modelo é treinado para produzir uma alta compatibilidade entre a *embedding* de uma imagem e a *embedding* da classe respetiva;
* O algoritmo percorre, por $T$ épocas, as $N$ instâncias de treino de modo a aprender as matrizes $W$ que melhor maximizam a compatibilidade entre cada par $(x_n,y_n)$.

## Experiências

* **Datasets**: CUB, Dogs e AwA

* ***Image embeddings***: *features* de 1024 dimensões extraídas do modelo pré-treinado GoogLeNet;
* ***Class embeddings***: atributos (binários e contínuos) - 85-dim, *word vectors* do word2vec, glove - 400-dim - e WordNet - 200-dim.

* As *features* das imagens foram normalizadas com o **z-score** e as *embeddings* das classe foram normalizadas segundo a norma $l2$;

* As matrizes $W$ foram inicializadas com valores aleatórios de média 0 e desvio-padrão $\frac{1}{\sqrt{d_x}}$;

* Após *cross-validation*, os parâmetros determinados foram os seguintes:
  * Nº de épocas: 150
  * *Learning rate (constante): $n_T = \{0.1,0.001,0.1\}$ para os *datasets* CUB, AwA e Dogs, respetivamente.
  * Escolha do $K$: $K \in \{2,4,6,8,10\}$

#### Resultados

* Top-1 accuracy (best): 
  * **CUB**: 32.5 % (Glove)
  * **AwA**: 71.9 % (Atributos)
  * **Dogs**: 25.2 % (WordNet)

* Os resultados podem ser melhorados combinando as diversas *embeddings* das classes, i.e., atributos+word2vec+glove+wordnet (no caso do AwA, a *accuracy* tem um acréscimo de 4%);

## Conclusões

* O LatEm é um método multimodal, uma vez que usa a imagens e informação auxiliar, ao nível das classes, anotada por humanos ou extraída automaticamente a partir de um *text corpus*;
* O LatEm incorpora múltiplas unidades de compatibilidade linear ($W$) e permite que cada imagem escolha uma delas;
* A aprendizagem do modelo é feita através de uma função objetivo baseada em *ranking* usando SGD.
