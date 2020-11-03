# A Survey of Zero-Shot Learning: Settings, Methods, and Applications
(http://www.ntulily.org/wp-content/uploads/journal/A_Survey_of_Zero-Shot_Learning_Settings_Methods_and_Applications_accepted.pdf)

## Introdução

* A maioria dos métodos de *machine learning* estão focados em classificar instâncias cujas classes já apareceram na fase de treino. Na prática, muitas das aplicações visam classificar instâncias cujas classes nunca tenham sido "vistas" anteriormente.

* No que diz respeito à aprendizagem supervisionada, a abordagem típica é treinar um modelo com um conjunto de dados de imagens contendo todas as classes que devem ser reconhecidas durante a fase de teste.

* Embora essa estratégia seja útil em um grande número de situações, existem vários cenários em que não é possível adquirir um número suficiente de amostras para todas as classes que se pretendem reconhecer. Nestes cenários, é desejável estender a aprendizagem supervisionada para cenários em que nem todas as classes estão disponíveis durante a fase de treino.

* O **Zero-Shot Learning (ZSL)** (também conhecido como *zero-data learning*) é um paradigma de aprendizagem poderoso e promissor, no qual as classes representadas pelas instâncias de treino e as classes que se pretendem classificar são disjuntas.

* O objetivo do ZSL é classificar instâncias pertencentes a classes que não têm instâncias etiquetadas, i.e., reconhecer objetos cujas instâncias não tenham sido "vistas" durante o treino do classificador.

* Revelando-se um método de aprendizagem promissor, o ZSL tem uma vasta gama de aplicações nas áreas da visão computacional, processamento linguagem natural e computação ubíqua.

#### Cenários de aplicação:

* Object recognition in computer vision;
* Fine-grained object classification.

Dado um conjunto de instâncias de treino pertencentes às classes conhecidas ($S$), o objetivo do ZSL é aprender um classificador que consiga classificar instâncias de teste pertencentes a classes desconhecidas.

A ideia geral do ZSL é, a partir do conhecimento contido nas instâncias de treino, classificar as instâncias de teste. Por esta razão, o ZSL é uma sub-área do *transfer learning*.

***Transfer Learning***: No *transfer learning*, o conhecimento contido no domínio fonte é transferido para o domínio alvo para aprendizagem do modelo.

Existem dois tipos de aprendizagem por transferência: *Homogeneous Transfer Learning* e *Heterogeneous Transfer Learning*, Neste último, o espaço de caraterísticas e o espaço de etiquetas são diferentes (ZSL);

**Auxiliary Information -> Semantic Information**

Nos trabalhos existentes, a abordagem de incluir informação auxiliar é inspirada na maneira como os humanos reconhecem o mundo à sua volta. Os humanos podem realizar a tarefa do ZSL com a ajuda de algum conhecimento semântico. Por exemplo, tendo o conhecimento de que uma zebra se parece com um cavalo e tem riscas, nós podemos reconhecer uma zebra mesmo sem nunca termos visto uma, sabendo apenas o que é um cavalo e o que são riscas. Desta forma, a informação auxiliar envolvida nos métodos de ZSL existentes é usualmente algum tipo de informação semântica.

**Induction**: o mesmo que aprendizagem supervisionada. O modelo é treinado com base num dataset com classes etiquetadas e esse modelo é usado para prever as classes de um dataset de teste. Modelo genérico.

**Transduction**: O treino é feito também tendo em conta o dataset de teste. No caso de aparecer um novo ponto na fase de treino, há necessidade de treinar novamente o modelo.

ZSL é categorizado em três tipos de configuração de aprendizagem:

1. _Class-Inductive Instance-Inductive (CIII)_ Setting - Apenas as instâncias de treino etiquetadas e os protótipos de classes conhecidas são usadas no treino do modelo.

2. _Class-Transductive Instance-Inductive (CTII)_ Setting - Instâncias de treino etiquetadas, protótipos de classes conhecidas e desconhecidas são usados no treino do modelo.

3. _Class-Transductive Instance-Transductive (CTIT)_ Setting - Instâncias de treino etiquetadas, protótipos de classes conhecidas e desconhecidas e instâncias de teste sem etiqueta são usados no treino do modelo.

**_Domain shift_**: Nos métodos de machine learning, dado que as distribuições das instâncias de treino e teste são diferentes, a performance do modelo treinado com as instâncias de treino irá decrescer quando aplicado às instâncias de teste. Este fenómeno é conhecido como *domain shift* e é mais crítico no caso do ZSL, dado que as classes abrangidas pelas instâncias de treino e de teste são disjuntas.

## Espaços Semânticos (*Semantic Spaces*)

Os espaços semânticos contêm informação semântica das classes e podem ser classificados quanto à sua forma de construção.

![](Imagens\Semantic_Spaces.PNG)

* **_Engineered Semantic Spaces_**

Nos espaços semânticos projetados cada dimensão é projetada por um humano.

  * _Attribute spaces_: compostos por um conjunto de atributos. São os mais usados no ZSL. Num espaço de atributos, os atributos são definidos como uma lista de termos que descrevem as propriedades da classe. Cada atributo é habitualmente uma palavra ou uma frase. Podem ser atributos reais (_continous attribute space_) ou binários (_binary attribute space_) ou ainda _relative attribute spaces_;
  * _Lexical Spaces_: compostos por um conjunto de items lexicais. São baseados nas etiquetas das classes e _datasets_ que possam fornecer informação semântica (WordNet).
  * _Text-keyword spaces_: compostos por um conjunto de palavras-chave extraídas de descrições textuais de cada classe.

A vantagem é a flexibilidade de codificar o domínio de conhecimento humano através da construção do espaço semântico e dos protótipos de classes. A desvantagem é a mão de obra humana requerida para a construção do espaço semantico.

* **_Learned Semantic Spaces_**

Neste tipo de espaços semânticos, os protótipos de cada classe são obtidos a partir do *output* de um qualquer método de *machine learning*.

  * _Label-embedding spaces_: os protótipos de classe são obtidos através da incorporação de etiquetas de classe. Usados em NPL; 
  * _Text-embedding spaces_: os protótipos de classe são obtidos incorporando as descrições textuais para cada classe. Por exemplo, numa tarefa de reconhecimento de imagens de objetos, muitas descrições textuais são coletadas para cada classe. Estas descrições textuais são usadas como _input_ de um _text encoder model_, e o _output_ é reconhecido como um protótipo de classe;
  * _Image-representation spaces_: os protótipos de classe são obtidos de imagens pertencentes a cada classe. 

A vantagem é que o processo de geração dos espaços semanticos é relativamente menos laboroso. A desvantagem é que os protótipos de classe são obtidos de modelos de *machine learning* e a semântica para cada dimensão está implícita. 

## Métodos

![](Imagens\ZSL_Methods.PNG)

Os métodos de ZSL podem ser classificados em duas categorias:

* **_Classifier-based methods_**: aprender um classificador para as classes não conhecidas;
* **_Instance-based methods_**: obter as instâncias etiquetadas pertencentes a classes não conhecidas e usá-las para aprendizagem do classificador.

#### _Classifier-based methods_

Os métodos baseados em classificadores, tipicamente, fazem uso de uma solução _One-vs-Rest_[^1] para aprender um classificador _zero-shot_ multi-classe $f^u(.)$, i.e., para cada classe desconhecida $c_i^u$, aprendem um classificador binário _one-vs-rest_: $f_i^u(.): \mathbb{R}^D \rightarrow \{0,1\}$, para a classe $c_i^u \in \upsilon$. Portanto, o eventual classificador _zero-shot_ $f^u(.)$ para as classes desconhecidas consiste em $N_u$ classificadores binários _one-vs-rest_ $\{f_i^u(.)|i=1,...,N_u\}$.

* **_Correspondence methods_**: a sua visão é construir o classificador para classes desconhecidas através da correspondência entre o classificador binário _one-vs-rest_ para cada classe e o protótipo de classe correspondente. 
  No espaço semântico, para cada classe, existe apenas um protótipo correspondente. Este protótipo pode ser reconhecido como a **representação** dessa classe. 
  Já no espaço de caraterísticas, para cada classe, há um classificador binário _one-vs-rest_ correspondente, que pode ser tido em conta como a **representação** dessa classe.
  Os métodos de correspondência têm por objetivo aprender a função correspondente (_correspondence function_) entre os dois tipos de **representação** referidos anteriormente.
  Nos métodos de correspondência, a função de correspondência $\varphi(.;\varsigma)$ tem como parâmetro de entrada o protótipo $t_i$ da classe $c_i$, e devolve o parâmetro $\omega_i$ do classificador binário _one-vs-rest_ $f_i(.;\omega_i)$ para esta classe, i.e. $\omega_i = \varphi(t_i;\varsigma)$. Depois de obter esta função de correspondência, para a classe desconhecida $c_i^u$, com o protótipo $t_i^u$, o correspondente classificador binário _one-vs-rest_ $f_i^u(.)$ pode ser construído.
  O procedimento geral para este tipo de métodos é o seguinte: primeiro, a função de correspondência $\varphi(.;\varsigma)$ é aprendida. De seguida, para cada classe desconhecida $c_i^u$, com o protótipo $t_i^u$ e a função de correspondência $\varphi(.;\varsigma)$, o classificador binário _one-vs-rest_ $f_i^u(.)$ é construído. Finalmente, com os estes classificadores binários $\{f_i^u(.)\}_{i=1}^{N_u}$ para as classes desconhecidas, a classificação de instâncias de teste $X^{te}$ é conseguida;

* **_Relationship methods_**: a sua visão é construir o classificador para as classes desconhecidas baseado nas relações entre as classes.
  No espaço de caraterísticas, classificadores binários _one-vs-rest_ $\{f_i^s(.)\}_{i=1}^{N_s}$ para as classes conhecidas podem ser aprendidos com os dados disponíveis. Entretanto, as relações entre as classes conhecidas e desconhecidas podem ser obtidas pelo cálculo das relações entre os protótipos correspondentes.
  Os métodos de relação têm por objetivo construir o classficador $f^u(.)$ para as classes desconhecidas através dos classificadores binários aprendidos para as classes conhecidas e as suas relações entre classes.
  O procedimento geral dos métodos de relação é o seguinte: primeiro, os classificadores binários _one-vs-rest_ $\{f_i^s(.)_{i=1}^{N_s}\}$ para as classes conhecidas $S$ são aprendidos com os dados disponíveis. De seguida, a relação $\delta$ entre as classes conhecidas e as classes desconhecidas é calculada correspondendo os protótipos de classe. Finalmente, com estes classificadores $\{f_i^s(.)\}_{i=1}^{N_s}$ e as relações $\delta$, o classificador $f^u(.) = \{f_i^u(.)\}_{i=1}^{N_u}$ para as classes desconhecidas $U$ é cosntruído e a classificação das instâncias de teste $X^{te}$ é conseguida.

* **_Combination methods_**: a sua visão é construir o classificador para as classes desconhecidas através da combinação de classificadores para elementos básicos que são usados para a constituição das classes.
  Nos métodos de combinação, é sabido que há uma lista de "_basic elements_" para constituir as classes. Cada uma das classes conhecidas e desconhecidas é uma combinação desses "_basic elements_". No espaço semântico, cada dimensão representa um _basic element_ e cada protótipo de classe indica a combinação desses _basic elements_ para a classe correspondente. Assim, os métodos nesta categoria são mais indicados para espaços semânticos onde cada dimensão dos protótipos de classe tenha o valor 0 ou 1, indicando se a classe tem o elemento correspondente ou não. É usado $a_i$ para indicar que o protótipo consiste em atributos.
  O procedimento geral dos métodos de combinação é a seguinte: com os dados disponíveis, os classificadores $\{f_i^a(.)\}_{i=1}^{M}$ para os atributos são aprendidos. Depois, com os classificadores aprendidos para os atributos, o classificador $f^u(.)=\{f_i^u(.)\}_{i=1}^{N_u}$ para as classes desconhecidas é obtido através de um qualquer modelo de inferência (framework).

[^1]: A estratégia _One-vs-Rest_ permite transformar um problema de classificação multi-classe num problema de classificação binária por classe. (Por exemplo, se tivermos as classes "red", "green" e "blue", então teremos 3 classificadores binários - um classificador binário para cada resultado possível.)


#### _Instance-based methods_

Os métodos baseados em instâncias visam obter primeiramente as instâncias etiquetadas para as classes desconhecidas e depois, com essas instâncias, aprender um classificador _zero-shot_ $f^u(.)$.

* **_Projection Methods_**: a sua visão é obter instâncias etiquetadas para classes desconhecidas projetando as instâncias do espaço de carateríticas e as instâncias do espaço semântico para um espaço comum (*projection space*). Neste sentido, podemos obter as instâncias etiquetadas pertencentes às classes desconhecidas.
  O procedimento geral dos métodos de projeção é o seguinte: inicialmente, as instâncias $x_i$ do espaço de caraterísticas $X$ e os protótipos $t_j$ do espaço semântico $T$ são projetados para o espaço de projeção $P$ com as funções de projeção $\theta(.)$ (pode ser uma função de regressão) e $\xi(.)$, respetivamente:

  $$
      X \rightarrow P:z_i = \theta(x_i) \\
      T \rightarrow P:b_j = \xi(t_j)
  $$

  Então, a classificação é feita no espaço de projeção, fazendo uso, por exemplo, do _nearest neighbor classification_ (1NN *classification*).

* **_Instance-Borrowing Methods_**: a sua visão é obter instâncias etiquetadas para as classes desconhecidas através do empréstimo de instâncias de treino.
  Os métodos baseados no empréstimo de instâncias são baseados nas similaridades entre classes. A ideia é utilizar o conhecimento de classes semelhantes àquela que se pretende classificar com o objetivo de reconhecer instâncias pertencentes às classes desconhecidas. 
  O procedimento geral destes métodos é o seguinte: inicialmente, para cada classe desconhecida $c_i^u$, algumas instâncias das instâncias treino são emprestadas e atribuidas à etiqueta da classe. Depois, com instâncias emprestadas para todas as classes desconhecidas, é feita a aprendizagem do classificador $f^u(.)$ para as classes desconhecidas e a classificação das instâncias de teste $X^{te}$ é conseguida.
  Nos métodos de empréstimo de instâncias, antes de acontecer o empréstimo, as classes desconhecidas devem ser determinadas. Apenas desta forma nós sabemos para que classes devemos emprestar as instâncias. Assim, a otimização do modelo é de maneira a pré-determinar as classes desconhecidas e naturalmente os protótipos das classes desconhecidas são envolvidas no processo de otimização.

* **_Synthesizing Methods_**: a sua visão é obter as instâncias etiquetadas para as classes desconhecidas através da síntese de algumas pseudo instâncias.
  O procedimento geral deste tipo de métodos é o seguinte: inicialmente, para cada classe desconhecida $c_i^u$, algumas pseudo instâncias etiquetadas são sintetizadas. Depois, com estas instâncias sintetizadas para todas as classes desconhecidas, o classificador $f^u(.)$ para as classes desconhecidas é aprendido e a classificação das instâncias de teste $X^{te}$ é conseguida.
  Tal como acontece com os métodos baseados no empréstimo de instâncias, antes de sintetizar as instâncias, as classes desconhecidas devem ser determinadas.


Categoria do Método | Vantagens | Desvantagens 
:-------------------|:----------|:------------
*Correspondence methods* | A correspondência entre os classificadores e os protótipos é capturada através de uma função de correspondência, que é simples e eficiente. | Não modelam o relacionamento entre as diferentes classes, que pode ser útil para o ZSL.
*Relationship methods* | Os relacionamentos entre as classes são modelados. Em alguns métodos, os classificadores para as classes conhecidas aprendidos noutros problemas podem ser diretamente utilizados. Desta forma, o custa da aprendizagem do modelo é reduzido. | O relacionamento entre as classes no espaço semântico é diretamente transferido para o espaço de caraterísticas. O problema de adaptação do espaço semântico para o espaço de caraterísticas é difícil de resolver.
*Combination methods* | Se existirem classificadores para a aprendizagem de atributos em outros problemas, podem ser usados diretamente. O custo de aprendizagem do modelo é reduzido. | As duas etapas de aprendizagem do classificador de atributos e da inferência a partir dos atributos para a classe é difícil de otimizar num processo unificado.
*Projection methods* | A escolha de funções de projeção é flexível e nós podemos escolher uma função apropriada de acordo com as caraterísticas do problema e do *dataset*. Especificamente, na aprendizagem CTIT, várias abordagens de aprendizagem semi-supervisionada podem ser adotadas nas etapas de projeção e classificação | Cada classe desconhecida tem apenas uma instância etiquetada (o protótipo). Por isso, os métodos de classificação (especialmente CIII e CTII) são usualmente limitados ao *nearest neighbor classification* ou métodos similares.
*Instance-borrowing methods* | O número de instâncias etiquetadas emprestadas às classes desconhecidas é relativamente grande. Então, vários modelos de classificação supervisionada podem ser usados. | As instâncias emprestadas são na verdade instâncias pertencentes às classes conhecidas. Isto causa incoerência aquando da aprendizagem do classificador de classes desconhecidas.
*Synthesizing methods* | O número de instâncias sintetizadas para as classes desconhecidas é relativamente grande e vários modelos de classificação supervisionada podem ser usados. | Assume-se que as instâncias sintetizadas seguem uma determinada distribuição (Gaussian distribution). Isto causa o viés das instâncias sintetizadas. 


##### *Multi-Label Zero-Shot Learning*

Há problemas em que cada instância pode ter mais do que uma etiqueta de classe associada, neste caso, estamos perante um problema que se enquadra no *Multi-Label ZSL*.

##### *Generalized Zero-Shot Learning*

No contexto do *Generalized Zero-Shot Learning* (GZSL), as instâncias de teste podem derivar de classes conhecidas e desconhecidas. No entanto, a performance deste método (GZSL) não é tão boa quanto a do tradicional ZSL.

  
## Aplicações

* **Visão Computacional**: aplicações relacionadas com imagens e vídeos. No que diz respeito ao reconhecimento de imagens, os problemas variam desde o reconhecimento de classes gerais de objetos, animais, até a uma granularidade mais fina, como por exemplo as o reconhecimento de espécies de pássaros e flores. Para além disso, o ZSL é também usado para segmentação de imagens, identificação de pessoas, recuperação de imagens, _domain adaptation_. Os _datasets_ mais usados incluem o AwA, AwA2, aPascal-aYahoo, ImageNet, CUB, Oxford Flower-102 e SUN Attribute. O ZSL é também largamente usado em problemas relacionados com vídeos, por exemplo, o reconhecimento de vídeos pertencentes a ações nunca antes vistas.

* **Processamento de linguagem natural**: Na área do processamento de linguagem natural (NLP), a aplicação do ZSL tem vinda a crescer nos últimos anos. É usado, por exemplo, para entendimento de linguagem falada (_spoken language classification_)

Para além destas duas áreas, o ZSL é também usado na área da computação ubíqua, para reconhecimento de atividade humana através de dados de sensores. 
Na área da biologia computacional é usado para análise de compostos moleculares, descodificação neuronal através de imagens de fMRI (_Functional magnetic resonance imaging_) e ECoG (_Electrocorticography_). 
Na área da segurança e privacidade é usado para reconhecimento de novos transmissores.

## Direções Futuras

* No reconhecimento de imagens de objetos, para além das caraterísticas de toda a imagem, caraterísticas sobre partes diferentes de objetos podem ser consideradas, igualmente;

* Exploração da heterogeneidade dos dados para treino e teste, i.e., o tipo de dados de treino e de teste não têm necessariamente de ser os mesmos;

 