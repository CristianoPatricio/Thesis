# Zero-Shot Learning with Semantic Output Codes

## Resumo

* Um classificador *Semantic Output Code* ([[SOC]]) usa o conhecimento base das propriedades semânticas das imagens a classificar para extrapolar para novas classes.
* Foi construído um **classificador SOC** para a tarefa de **descodificação neuronal**, i.e., a capacidade de **prever palavras** que as pessoas estão a pensar, **a partir da análise de imagens de ressonâncias magnéticas** (fMRI), mesmo sem exemplos de treino para essas palavras.


## Introdução

* O ZSL é um problema interessante, especialmente no domínio em que o conjunto de imagens a classificar padece de milhares de classes diferentes. No caso concreto da Visão Computacional, onde há centenas de milhares de objetos que queremos que o computador seja capaz de reconhecer;
* A descodificação da atividade neuronal, onde o objetivo é determinar a palavra ou o objeto que a pessoa está a pensar, a partir da análise de uma imagem da atividade cerebral dessa pessoa;
  
A questão geral deste *paper* é:
> Dada a codificação semântica de um grande conjunto de classes, podemos construir um classificador para reconhecer classes que são omitidas do conjunto de treino?

## Classificação com Conhecimento Semântico

* O objetivo não é atribuir uma unica designação a uma classe, mas sim um conjunto possível de *features* semânticas que caraterizem um grande número de classes possíveis;
* O modelo irá aprender relações entre os dados de entrada e as *features* semânticas;
* Dado um novo *input*, o modelo irá prever um conjunto de *features* semânticas correspondentes a esse *input*, e depois encontrar a classe no conhecimento base que melhor se relaciona com o conjunto das *features* previstas;

Um classificador SOC $H:X^d \rightarrow Y$ mapeia pontos de qualquer espaço d-dimensional para uma *label* do conjunto $Y$, tal que $H$ é a composição de duas funções, $S$ e $L$, tal que:

$$
\begin{matrix}
 H = L(S(.)) \\ S : X^d \rightarrow F^p \\ L : F^p \rightarrow Y
\end{matrix}
$$

Por exemplo, dada uma imagem de um cão, esta é mapeada primeiramente para o espaço semântico e só depois é que é mapeada para a class "cão". Como resultado, as ***labels* das classes** podem ser vistas como um ***semantic output code***.

* O SOC é um processo composto por duas fases: (1) a primeira fase $S(.)$ é a coleção dos classificadores lineares (um classificador por *feature*) e a segunda fase $L(.)$ é um classificador 1NN usando a métrica de distância *Hamming*.

* Para **aprender** o mapeamento **$S$**, o classificador contrói um conjunto de **$N$ exemplos $\{x,f\}_{1:N}$**, substituindo cada $y$ pelo respetivo codificador semântico $f$ de acordo com o conhecimento base $k$;
  
* Quando um novo exemplo é apresentado, o classificador irá fazer a previsão do codificador semântico usando o mapeamento $S$ que foi aprendido. Mesmo quando um novo exemplo pertence a uma classe que não aparece no conjunto de treino, se a previsão produzida por $S$ está perto da codificação verdadeira da classe, então o mapeamento $L$ terá uma oportunidade de classificar com a *label* correta. Por exemplo, se o modelo consegue prever que o objeto tem pêlo e uma cauda, então há uma grande probabilidade da classe ser "cão", mesmo sem nunca ter visto imagens de cães durante o treino.

> Usando uma codificação rica das classes, o classificador poderá estar apto a extrapolar e a reconhecer novas classes

## Análise Teórica

##### Sob que condições o classificador SOC reconhecerá exemplos de classes omitidas no conjunto de treino?

Para responder a esta questão, o objetivo é obter um limite probably approximately correct (PAC): saber que quantidade de erro pode ser tolerado na previsão de propriedades semânticas aquando da recuperação da classe nova com grande probabilidade.
Este erro será usado para obter um limite no número de exemplos necessários para conseguir este nível de erro na primeira fase do classificador. A ideia é que na primeira fase ($S(.)$) do classificador, possa prever bem as propriedades semânticas, depois, na segunda fase ($L(.)$) haverá uma grande probabilidade de recuperar a *label* correta das novas classes.

* Assumimos que as *features* semânticas são *labels* binárias.

* A tolerância para o erro é relativa a um ponto particular em relação a outros pontos no espaço semântico.

## Caso de Estudo: Descodificação Neuronal de Novos Pensamentos

O objetivo é decodificar palavras novas que uma pessoa esteja a pensar, através de imagens de ressonâncias magnéticas da atividade neuronal da pessoa, sem incluir, contudo, imagens dessas palavras durante a fase de treino.

##### Datasets
* ***fMRI dataset***: este dataset contém a atividade neuronal observada de nove pessoas enquanto viam 60 palavras diferentes (5 exemplos para 12 categorias diferentes).
* Este *dataset* representa os dados de treino;
* Foram recolhidas 2 bases de conhecimento semântico: *corpus5000* e *human218*;
  
##### Modelo

Foi usada regressão linear com outputs multiplos para aprender o mapeamento $S$;

$$
\hat{W} = (X^TX+\lambda I)^{-1}X^TY
$$

onde $I$ é a matriz identidade e $\lambda$ é um parâmetro de regularização. 

> Dada uma nova imagem fMRI $x$, podemos obter uma previsão $\hat{f}$ das *features* semânticas para essa imagem, multiplicando a imagem pelos pesos: $\hat{f} = x.\hat{W}$.

Para a segunda etapa do classificador SOC, $L(.)$, é usado um classificador 1-nearest neighbor. Por outras palavras, $L(\hat{f})$ devolve o ponto mais próximo numa dada base de conhecimento de acordo com a distância Euclidiana.

## Experiências

> Será possível construir um classificador para distinguir entre duas classes, onde essas duas classes não aparecem no conjunto de treino?

* Foi treino o modelo da Equação 3 para aprender o mapeamento entre 58 imagens fMRI e as *features* semânticas para as respetivas palavras. Para a primeira imagem apresentada, foi aplicada a matriz de pesos aprendida para obter a previsão das *features* semânticas e depois usado o classificador 1NN para comparar o vetor das previsões às codificações semânticas verdadeiras das palavras. A *label* foi escolhida selecionando a palavra com a codificação mais próxima à da previsão para imagem fMRI.
* As *features* semânticas **human218** superam as **corpus5000**;
  

## Conclusões

* Foi apresentado um classificador SOC para o problema do ZSL;
* O classificador é capaz de prever novas classes, que foram omitidas no conjunto de treino, considerando apenas o conhecimento semântico base usado para codificar as *features* das imagens que são apresentadas. Esta codificação é depois comparada com as codificações do espaço semântico, de forma a prever a nova classe.





