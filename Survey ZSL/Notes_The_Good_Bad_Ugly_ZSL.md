# Zero-Shot Learning: A Comprehensive Evaluation of the Good, the Bad and the Ugly

O ZSL tem por objetivo o reconhecimento de objetos cujas instâncias não tenham sido "vistas" durante a fase de treino.

Todos os anos têm sido propostos novos métodos de ZSL (*good aspects*), contudo é difícil quantificar o progresso pelo facto de que não existe um protocolo de avaliação estabelecido (*bad aspects*). De facto, a procura pela melhoria dos números levou a que surgissem protocolos de avaliação falhados (*ugly aspects*).

O cerne da questão de todos os métodos de ZSL é associar classes observadas e não-observadas através de informação auxiliar, que codifica visualmente propriedades distinguíveis dos objetos.

A "per-class averaged top-1 accuracy" é uma métrica de avaliação importante quando o dataset não é tão balanceado no que diz respeito ao número de imagens por classe.

Os métodos de ZSL devem ser avaliados em classes raras ou menos povoadas.

## Related Work

Trabalhos recentes de ZSL fazem uso dos atributos numa abordagem de 2-fases para inferir a etiqueta de uma imagem que pertence a uma classe desconhecida. 

Em geral, os atributos de uma imagem de input são estimados numa primeira fase, só depois é que a etiqueta da classe é inferida procurando a classe que contém o conjunto de atributos mais similares.

Os modelos de duas fases têm o problema do *domain shift* entre a tarefa intermediária e a tarefa final, i.e., a tarefa final é prever a etiqueta de class e a tarefa intermediária é a aprendizagem dos classificadores de atributos.

Estudos recentes em ZSL aprendem diretamente o mapeamento do espaço de caraterísticas da imagem para o espaço semântico.

No ZSL, qualquer forma de informação secundária (atributos - propriedades visuais e partilhadas dos objetos) é necessária para partilhar informação entre classes para que o conhecimento aprendido das classes conhecidas seja transferido para as classes desconhecidas.

O ZSL tem sido criticado por ser um set-up restritivo na medida em que a imagem usada na fase de predição tem ser apenas relativa a classes desconhecidas. Então, o GZSL foi proposto para generalizar a tarefa do ZSL para o caso em que as classes conhecidas e desconhecidas são usadas no momento de teste.

O artigo [63] fornece uma comparação da avaliação de 5 métodos em 3 datasets incluindo o ImageNet com 3 divisões standard e propõe uma métrica para avaliar a performance do GZSL.

## Evaluated Methods

No momento do teste, na definição ZSL, o objetivo é atribuir uma classe desconhecida a uma imagem de teste. Já no GZSL, a imagem de teste pode ser atribuida a uma classe conhecida ou desconhecida, que tenha o maior valor de compatibilidade.

***Learning Linear Compatibility***

(Métodos: ALE, DEVISE, SJE, ESZSL, SAE)

Dado uma imagem, as frameworks de aprendizagem de compatibilidade prevêem a classe que atinge o score máximo de compatibilidade com a imagem.

ESZSL [10]: A vantagem desta abordagem é que a função objetivo é convexa e é uma solução *closed-form*.

***Learning Non-Linear Compatibility***

(Métodos: LATEM, CMT)

***Learning Intermediate Attribute Classifiers***

(Métodos: DAP, IAP)

DAP [1] aprende classificadores probabilisticos de atributos e efetua a previsão da classe combinando os scores dos classificadores de atributos aprendidos.

Uma nova imagem é atribuida a uma das classes desconhecidas usando:

$$
f(x) = \argmax_x \prod_{m=1}^{M}\frac{p(a_m^c|x)}{p(a_m^c)}
$$

* $M$ é o número total de atributos
* $a_m^c$ é o m-ésimo atributo da classe $c$
* $p(a_m^c|x)$ é a probabilidade do atributo dada a imagem $x$
* $p(a_m^c)$ é o atributo estimado através da média empírica sobre as classes de treino.

São treinados classificadores binários (*logistic regression*) que dão os resultados de probabilidade dos atributos com respeito às classes de treino.

***Hybrid Models***

(Métodos: SSE, CONSE, SYNC, GFZSL)

*Semantic Similarity Embedding* (SSE) [13], *Convex Combination
of Semantic Embeddings* (CONSE) [15] e os *Synthesized
Classifiers* (SYNC) [14] expressam as imagens e as classes semânticas incorporadas com uma mistura de proporções de classes conhecidas, por isso são chamados de modelos híbridos.

***Transductive ZSL Setting***

Esta abordagem do ZSL implica que imagens sem etiqueta de classes desconhecidas estejam disponíveis durante o treino. Usando imagens sem etiqueta é esperado uma melhoria da performance uma vez que essas imagens contêm informação latente das classes desconhecidas.

## Datasets

SUN, CUB, AWA1, AWA2 and aPY

A desvantagem do AWA1 é que as imagens não estão disponíveis publicamente, o que levou à introdução do AWA2, que conta com mais 6827 imagens que o seu antecessor. Comparado com o AWA1, o AWA2 contém, portanto, mais imagens: cavalos e golfinhos nas classes de teste e antílopes e vacas nas classes de treino.

*t-distributed stochastic neighbor embedding (t-SNE) is a machine learning algorithm for visualization based on Stochastic Neighbor Embedding*

## Protocolo de Avaliação

#### *Image and Class Embedding*

Foi realizada a extração de caraterísticas das imagens, denominadas *image embeddings*  (2048-dim top-layer pooling units da ResNet-101), de toda a imagem de todos os datasets.

Para além das caraterísticas das imagens, também os atributos de classes são importantes, tendo sido utilizados valores binários (0 e 1).

#### *Evaluation Criteria*

A accuracy de uma única classificação de uma imagem etiquetada tem sido calculada com o Top-1 accuracy, i.e., a previsão é precisa quando a classe prevista é a correta.

Se a accuracy for a média para todas as imagens, uma performance maior será conseguida para as classes populadas.

Desta forma, é feita a média das previsões corretas independemente para casa classe antes de dividir a sua soma acumulada com respeito ao número de classes. 

A fórmula da *average per-class top-1 accuracy* é calculada da seguinte forma:

$$
acc_y = \frac{1}{\left \| Y \right \|} \sum_{c=1}^{\left \|  Y \right \|} \frac{{\#correct \, predictions \, in \, c}}{{\#samples \, in \, c}}
$$

No caso do GZSL, dado que o espaço de pesquisa no momento da avaliação não é restrito às classes de teste, uma vez que também inclui classes de treino, é necessário calcular a média harmónica das acc de treino e de teste:

$$
H = \frac{2*{acc}_{y^{tr}}*{acc}_{y^{ts}}}{{acc}_{y^{tr}}+{acc}_{y^{ts}}}
$$

Nota: A média harmónica é um tipo de média que é geralmente utilizada em situações em que é desejável o cálcula da média envolvendo duas ou mais taxas.

Foi utilizada a média harmónica como critério de avaliação porque no caso da média aritmética, se a accuracy da classe conhecida for muito alta, afeta o resto dos resultados significativamente. Contrariamente, o objetivo é uma accuracy alta em ambas as classes: conhecidas e desconhecidas.

## Experiências

#### ZSL Setting

É concluido que melhorando as caraterísticas visuais leva a uma melhoria nos resultados do ZSL.

É também evidenciado que o método GFZSL é o que mais vezes ocupa o primeiro lugar no *ranking*.

Concluiu-se que os métodos GFZSL e ALE parecem ser os métodos mais robustos na abordagem ZSL clássica para datasets de atributos.

Os três métodos mais "fraquinhos" são o IAP, o CMT e o CONSE.

Através de uma análise cuidadosa, relativamente à performance do método SJE, o mesmo seleciona hiperparametros diferentes para cada uma dos datasets (AWA1 e AWA2), o que causa resultados ligeiramente diferentes. É por isso concluido que o ZSL é sensível à afinação de parâmetros.

Os resultados da avaliação cruzada-dataset indica que o AWA2 é um bom substituto do AWA1.

**Resultados do ZSL no ImageNet**

O método que demonstrou melhor performance foi o SYNC, o que pode indicar que se comporta bem numa aborgadem de grande escala e pode aprender em cenários incertos devido ao uso do Word2Vec ao invés de atributos. O Word2Vec pode ser afinado para o SYNC, como é relato por vários autores.

Por outro lado, o GFZSL, que se mostrou o melhor método nos datasets de atributos, obteve um desempenho muito baixo no ImageNet, o que leva a crer que os modelos generativos requerem uma quantidade muito grande de atributos para executar bem a tarefa do ZSL.

O segundo método com melhor desempenho foi o ESZSL, que tem um mecanismo de regularização. Foi também observado que o tipo de classes presentes no conjunto de teste é mais importante que o número de classes. Por isso, a seleção do conjunto de teste é um aspeto importante do ZSL em datasets de grande escala.

**Resultados do *Generalized Zero-Shot Learning***

Os resultados do GZSL são significamente piores do que os resultado do ZSL. Isto porque as classes de treino são incluídas no espaço de pesquisa, atuando como "distractors" para as imagens que vêm das classes de teste.

As frameworks ALE, DEVISE e SJE tem melhor desempenho nas classes de testes, enquanto que os métodos que aprendem atributos independentes os classificadores de objetos, DAP e CONSE, têm melhor desempenho nas classes de treino.

Similarmente aos resultados do ImageNet, o GFZSL tem pior desempenho na abordagem GZSL.

Resumidamente, o GZSL fornece mais um nível de detalhe no desempenho dos métodos do ZSL. A accuracy das classes de treino é tão importante quanto a accuracy das classes de testes em cenários do mundo real. Os métodos devem ser desenhados de maneira que eles estejam aptos a prever tão bem nas classes de treino e de teste.

***Transductive (Generalized) Zero-Shot Learning***

As abordagens transdutivas usam imagens sem etiqueta das classes de teste durante a fase de treino.

* **3 abordagens state-of-the-art de abordagens transductive ZSL:**
  * DSRL
  * GFZSL-tran
  * ALE-tran

Na abordagem ZSL, a aprendizagem transdutiva leva a uma melhoria do valor da accuracy.

O GFZSL-tran e o ALE-tran superam os originais GFZSL e o ALE em todos os casos testados.

## Conclusão

* **Métodos ZSL**
  * [1] - *Direct Attribute Prediction* (DAP); *Indirect Attribute Prediction* (IAP)
  * [7] - *Deep Visual Semantic Embedding* (DEVISE)
  * [9] - *Structured Joint Embedding* (SJE)
  * [10] - ESZSL
  * [11] - *Latent Embeddings* (LATEM)
  * [12] - *Cross Modal Transfer* (CMT)
  * [13] - *Semantic Similarity Embedding* (SSE)
  * [14] - *Synthesized Classifiers* (SYNC)
  * [15] - *Convex Combination of Semantic Embeddings* (CONSE)
  * [30] - *Attribute Label Embedding* (ALE), (ALE-tran)
  * [33] - (SAE)
  * [41] - (GFZSL), (GFZSL-tran)
  * [71] - (DSRL)

Foi observado que os dados sem etiqueta de classes desconhecidas podem melhorar mais os resultados do ZSL, por isso não justo comparar abordagens de aprendizagem transductiva com as abordagens indutivas.

Foi ainda descoberto que algumas divisões standard dos dataset ZSL tratam aprendizagem de caraterísticas de forma diferente na fase de treino, uma vez que muitas das classes de teste estão incluídas no ImageNet1K que é usado para treinar DNN que atuam como feature extrators. Desta forma, foi proposto um novo método de divisão que garante que nenhuma das classes de teste pertence ao ImageNet1K. Além do mais, divisões disjuntas de classes de treino e de validação é um componente necessário na afinação de parâmetros na abordagem ZSL.

Finalmente, a inclusão de classes de treino no espaço de pesquisa, i.e., GZSL, pode ser uma área interessante para investigação futura.

Foi proposta também a média harmónica como medida de performance na abordagem GZSL.


























