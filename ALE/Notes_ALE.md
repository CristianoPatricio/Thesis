# Label-Embedding for Attribute-Based Classification

## Resumo
* Cada **classe** é incorporada no **espaço dos vetores de atributos**, dando o nome ao método *Attribute Label Embedding* (**ALE**);
* São aprendidos os parâmetros de uma função que mede a compatibilidade entre uma imagem e a *embedding* da *label*;
* É através dos parâmetros aprendidos que é possível associar um vetor de *features* de uma dada imagem a um vetor de atributos no espaço dos atributos;

## Introdução
* Uma das soluções para o problema do ZSL é a introdução de uma camada intermédia de atributos;
* **Os atributos correspondentem a propriedades de alto nível dos objetos e podem sãer partilhados entre diversas classes**;
* O tradicional algoritmo para classificação baseada em atributos necessita de aprender um classificador por cada atributo. Para classificar uma imagem, primeiro são previstos os atributos através dos classificadores aprendidos e os *scores* dos atributos são combinados de forma a determinar a classe. Esta estratégia é conhecida como *Direct Attribute Prediction* (**DAP**).
* Apesar dos atributos serem uma fonte de informação útil, outras f**ontes de informação** podem ser alavancadas para o ZSL, como por exemplo informação semântica hierarquica, como o **Wordnet**.
* É introduzida uma **função** que mede a **compatibilidade** entre uma **imagem** $x$ e uma ***label*** $y$;
* Os parâmetros da função são aprendidos num conjunto de treino para garantir que, dada uma imagem, as classes corretas têm maior *ranking* que as incorretas. **Dada uma imagem de teste, o reconhecimento consiste na procura da classe com a maior compatibilidade**;

## Modelo

$$
f(x;w) = \argmax_{y\in Y}F(x,y;w)
$$

onde $w$ é o vetor de parâmetros do modelo $F$ e $F$ mede o quão compatível é o par $(x,y)$, dado $w$.

## Experiências

* As experiências foram realizadas tendo por base os *datasets* AwA e CUB;
* Os resultados são reportados segundo a métrica **Top-1 accuracy**;

## Datasets

* *Animals With Attributes* (**AwA**) e *Caltech-UCSD-Birds* (**CUB**)

## In a nutshell

Este método faz uso de uma função que aprende a medir a compatibilidade entre uma imagem $x$ e a classe $y$. A classe atribuida à imagem de teste é aquela que maior compatibilidade terá com a imagem de teste $x$.