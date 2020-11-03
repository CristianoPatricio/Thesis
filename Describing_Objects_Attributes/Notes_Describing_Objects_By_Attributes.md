# Describing Objects by their Attributes

## Resumo

* A descrição de objetos através de atributos permite o reconhecimento de novos objetos com poucos ou nenhuns exemplos visuais;

* A aprendizagem de atributos apresenta um novo grande desafio: a generalização através das categorias de objetos;

* É apresentado um método para aprendizagem de atributos que consegue generalizar bem através de categorias.
  
* Neste paper é apresentada uma abordagem baseada em atributos não só para reconhecer categorias de objetos, mas também para descrever objetos desconhecidos, reportar atributos atípicos nas classes conhecidas e mesmo aprender modelos de novas categorias de objetos a partir da descrição textual.

![Abordagem](Imagens/Método_Describing_Objects_By_Their_Attributes.PNG)

## Introdução

* A capacidade de inferir atributos permite-nos descrever, comparar e categorizar objetos mais facilmente. Quando confrontados com um novo tipo de objeto, podemos dizer algo sobre ele (e.g., "dog with spots") e aprender a reconhecer objetos a partir da sua descrição;
* O foco é aprender atributos de objetos. Podem ser aprendidos a partir de anotações, o que permite depois identificá-los com base nas descrições textuais;
* Muitas vezes, essas anotações não são suficientes para diferenciar duas categorias de objetos, pelo que é também fundamental aprender atributos não-semânticos, que correspondem a partições no espaço de caraterísticas visuais.
* É proposto que sejam primeiramente extraídas caraterísticas que podem ser usadas para prever atributos numa classe de objetos e usar essas caraterísticas para treinar os classificadores de atributos;
  
## Atributos e Caraterísticas

* A inferências de atributos de objetos é o problema chave no reconhecimento;
* Os atributos semânticos são, por exemplo, partes, formas, materiais. Mas muitas vezes estes atributos não chegam para distinguir todas as categorias de objetos. Por essa razão são usados os atributos discriminativos, que tomam a forma de comparações: "os gatos e os cães têm isto, mas as ovelhas e os cavalos não têm.";
* Aprendendo atributos semânticos e discriminativos, podemos reconhecer objetos usando os atributos estimados como caraterísticas, mas também descrever objetos que não nos são familiares;

### Caraterísticas Base (*Base Features*)

* A grande variedade de atributos requer uma representação de caraterísticas para descrever vários aspetos visuais. São usadas a cor e a textura, que são boas para os materiais; palavras visíveis, que são úteis para partes; e arestas que são úteis para formas. São as chamadas *base features*.

### Atributos Semânticos

* São usados 3 tipos de atributos semânticos:
  * ***Shape** attributes*: referem-se às propriedades 2D e 3D;
  * ***Part** attributes*: identificam as partes visíveis ("has head", "has leg", "has arm", "has window");
  * ***Material** attributes*: descrevem de que mateiral é feito o objeto ("has wood", "is furry"; "has glass", "is shiny");

### Atributos Discriminativos

* As instâncias relativas a cães e gatos partilham dos mesmos atributos semânticos, o que causa que o sistema reconheça cada um deles apenas com uma *accuracy* de 74\%. Para resolver este problema, são introduzidos atributos discriminativos auxiliares;
* Estes novos atributos tomam a forma de comparações aleatórias. Cada comparação divide uma porção de dados em duas partições. Estas partições são formadas selecionando aleatoriamente de uma a cinco classes or atributos para cada lado. Por exemplo, uma divisão coloca de um lado o "cão" e de outro lado o "gato". Cada divisão é definida por um subconjunto de *base features*, como por exemplo, textura ou cor. Se for escolhida a textura para distinguir entre cães e gatos, usamos uma SVM linear para aprender dezenas dessas divisões e escolher aquela que melhor prevê usando os dados de validação.

#### Seleção de Caraterísticas

* A seleção de caraterísticas é fundamental para evitar confudir os classificadores. Por exemplo, se nós quisermos aprender um classificador de "rodas", nós selecionamos caraterísticas que se comportem bem em distinguir exemplos de carros com "rodas" e carros sem "rodas". A seleção de caraterísticas é feita usando um regressor logístico com regularização L1, treinado para cada atributo de classe.

## Experiências

* Os testes demonstraram que a seleção de caraterísticas fornece avultadas melhorias na aprendizagem a partir de descrições textuais;
* É possível classificar objetos apenas a partir da sua descrição textual;
* A performance na tarefa de nomear os objetos foi comparada com duas bases de referência: SVM linear e regressão logística aplicada às caraterísticas base para reconhecer diretamente os objetos;

##### Aprender para identificar novos objetos
* São usados os atributos estimados como caraterísticas e um SVM linear *one-vs-all* como classificador.

##### Aprender novas categorias a partir de descrição textual
* Por exemplo, podemos aprender novas categorias descrevendo novas classes para o nosso algoritmo, i.e., a nova classe é "furry", "four legged", "has snout" e "has head". 
* A descrição do objeto é especificada por uma lista de atributos, fornecendo um vetor de atributos binário. A classificação da imagem de teste é feita encontrando a descrição mais próxima aos atributos previstos;
* Aprender novas categorias a partir de descrições textuais tem uma *accuracy* baixa, 32.5\%;

##### Rejeição
* Quando um objeto de uma nova categoria é apresentado ao modelo, queremos que o nosso modelo reconheça que o objeto não pertence a nenhuma categoria conhecida.

## Conclusão

* Inferir propriedades de objetos deve ser um objetivo no reconhecimento de objetos;
* Aprender atributos permite várias habilidades:
  * Prever propriedades de novos tipos de objetos;
  * Identificar informação atípica sobre um objeto familiar;
  * Aprender a partir de descrições textuais;



