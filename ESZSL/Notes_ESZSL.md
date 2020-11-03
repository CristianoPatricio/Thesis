# An embarrassingly simple approach to zero-shot learning

## Introdução

* As abordagens de classificação de objetos no paradigma da aprendizagem supervisionada não consegue lidar com situações onde, na fase de teste, apareçam imagens pertencentes a classes que não foram treinadas;
* O ZSL vem resolver este problema;
* O ZSL consiste no reconhecimento de novas categorias de instâncias sem exemplos de treino, tendo por base uma descrição de alto nível das novas categorias, que as relaciona com as categorias aprendidas pelo modelo.
* O objetivo do *domain adaptation* é aprender uma função a partir de dados de um domínio para que possa ser aplicada a dados de um domínio diferente.

## Modelo

* Abordagem assente em duas camadas para modelar o relacionamento entre *features*, atributos e classes com a ajuda de um modelo linear;
* A primeira camada contém os pesos que descrevem o relacionamento entre as *features* e os atributos, e é aprendida na fase de treino;
* A segunda camada modela o relacionamento entre os atributos e as classes;

## Experiências

* O **método** foi **testado** nos *datasets*  **AwA**, **SUN** e **aPY**;
* No *dataset* AwA, cada classe é descrita por um conjunto de atributos, já os *datasets* SUN e aPY, a anotação dos atributos é feita por imagem, logo há necessidade de fazer a média dos atributos pertencentes à mesma classe para converter a anotação em *per class*;
* O ESZSL supera o DAP em toda a gama de número de classes;
* Em comparação com o DAP, que demora cerca de 11 horas para treinar 2000 instâncias, o ESZSL demora apenas 4.12 segundos!!
* A principal diferença do ESZSL para o DAP é que o ESZSL faz uso de um regularizador mais elaborado;
* 