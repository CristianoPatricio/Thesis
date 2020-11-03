# Zero-Shot Learning Through Cross-Modal Transfer

## Resumo

* É apresentado um modelo que consegue reconhecer objetos em imagens mesmo que não estejam disponíveis nenhuns dados de treino para a classe do objeto;
* O único conhecimento necessário acerca das categorias desconhecidas é proveniente de informação semântica (*text corpora*);

## Introdução

* É mostrado, neste trabalho, como fazer uso do vasto conhecimento existente sobre o mundo visual disponível na linguagem natural para classificar objetos nunca vistos pela rede;
* O modelo será capaz de prever classes conhecidas e desconhecidas;

Em primeiro lugar, as imagens são mapeadas para um espaço semântico de palavras que é aprendido por um modelo NN. Em segundo lugar, como o modelo inclui um método de "deteção de novidades", a nova imagem será avaliada para determinar se pertence às classes conhecidas ou não. Se a imagem é uma categoria conhecida, então um classificador standard pode ser usado. Se não, as imagens são associadas a uma classe com base na proximidade com os vetores das classes desconhecidas. **Este detetor de novidade serve, no fundo, para impedir que imagens que nunca tenham sido vistas durante o treino possam ser associadas a classes de treino.**

* São exploradas duas estratégias para a deteção de novidades: (1) preferência por uma *accuracy* alta para as classes desconhecidas e (2) preferência por uma *accuracy* alta para as classes conhecidas;

### Representações das Imagens e das Palavras
* Os *word vectors* são inicializados e representados por *word vectors* pré-treinados de 50 dimensões;
* As imagens são representadas por um vetor com $I$ *features* segundo o método supervisionado de Coates et al. [6];

### Projeção das Imagens para o Espaço de *Embeddings* Semântico
* Todas as imagens de treino $x$ de uma classe conhecida $y$ são mapeadas para o vetor de palavra $w_y$ correspondente ao nome da classe. Para treinar este mapeamento, foi treinada uma 2-layer NN para minimizar o função objetivo:

$$
J(\Theta) = \sum_{y \in Y_s} \sum_{x^{(i)}\in X_y} \left | \left | w_y - \theta^{(2)}f(\theta^{(1)}x^{(i)}) \right | \right |^2
$$

onde $f = tanh$ e $\Theta = (\theta^{(1)}, \theta^{(2)})$. A função de custo foi treinada com a back-prop standard L-BFGS.

### Modelo ZSL

* O objetivo é prever $p(y|x)$, para ambas as classes $y \in Y_s \cup Y_u$ conhecidas e desconhecidas, dada uma imagem de teste $x$.
* É introduzido um detetor de novidade (**Gaussian threshold model**; **LoOP model**), variável binária $V \in \{s,u\}$, que indica se a imagem de teste pertence a uma classe conhecida ou desconhecida;
* Uma imagem pertencente a uma classe desconhecida não estará muito próxima das imagens de treino, contudo, estará na mesma região semântica;
* Na fase de teste, pode ser usado um ***outlier detection method* para determinar se é uma imagem de uma classe conhecida ou desconhecida**;
* No fundo, o método faz uso de um ***threshold* para delimitar a área a considerar para o cálculo do valor de probabilidade da classe ser conhecida ou desconhecida**;
* ***Thresholds*  mais pequenos significa que menos imagens possam vir a ser consideradas como pertencentes a classes desconhecidas**;
* In order to get some belief about the outlierness of a new test image, the ***Local Outlier Probability*** can be used to weight the seen and unseen classifiers.

#### Classificação

No caso de o detetor de novidade detetar que a classe da imagem de teste é conhecida, então é usado um classificador *standard* para classificar a imagem, por exemplo, um classificador **softmax**. Caso contrário, se o detetor de novidade detetar que a classe é desconhecida, assumimos uma distribuição Gaussiana isométrica em torno de cada *word vector* da nova classe e atribuimos as classes com base na sua probabilidade, considerando os *word vector* que estão mais próximos.

## Experiências

Para a maioria das experiências foi o usado o *dataset* **CIFAR-10**.

* O **CIFAR-10** é constituído por 10 classes, cada uma delas com 5000 imagens de 32x32x3 RGB;
* Foi usado o método[^1] não-supervisionado de extração de caraterísticas para obter um vetor de *features* de 12800 dimensões para cada imagem;
* Os vetores de palavras são proveninentes do *dataset* Huang[^2] e têm 50 dimensões. Cada vetor corresponde uma categoria do CIFAR;
* Durante o treino, foram omitidas 2/10 classes, que foram reservadas para a tarefa de *zero-shot*;

##### Avaliação das Classes Conhecidas

* Em primeiro lugar foram avaliadas os resultados das classes separadamente. Para as 8 classes conhecidas foi treinado um classificador softmax, que obteve uma *accuracy* de 82.5\% na previsão das classes verdadeiras;
* No caso da avaliação das duas classes remanescentes, nunca treinadas, a classificação foi baseada na isometria Gaussiana, que compara as distâncias entre os vetores de palavras das classes desconhecidas e a imagem mapeada para o espaço semântico. Neste caso, o valor da *accuracy* é tanto maior quanto a distância semântica entre duas classes, p.e., gatos e camiões.

##### Avaliação das Classes Desconhecidas
* A *accuracy* máxima é obtida quando as classes escolhidas para classificação são pertencentes a categorias distintas semânticamente;

##### Influência dos Detetores de Novidade 
* O detetor de novidade decide se uma dada imagem de teste pertence a uma classe que foi usada para treino ou não. Dependendo da escolha, a imagem passa num classificador **softmax** no caso de ser uma classe conhecida ou então, caso contrário, é atribuída à classe representada pelo *word vector* semâmntico mais próximo das classes desconhecidas;
* O modelo **LoOP** dá-nos melhores valores de *accuracy* que o modelo **Gaussian**;
  
##### Comparação com a Classificação baseada em Atributos
* Em geral, a vantagem do método proposto neste paper, é a habilidade para a adaptação ao domínio, que é difícil no caso da classificação baseada em atributos, na medida em que os atributos apropriados necessitam de ser esoclhidos cuidadosamente.

## Conclusão
* Foi introduzido um novo modelo que reune classificação *standard* e *zero-shot* baseada na representação profunda de imagens e palavras;
* As duas ideias chave são: (1) usando representações semânticas de vetores de palavras pode ajudar na transferência de conhecimento entre modalidades mesmo quando essas representações são aprendidas de uma maneira não supervisionada e (2) a framework Bayesian que faz a distinção entre as novas classes desconhecidas e os pontos na espaço semântico das classes de treino pode ajudar a combinar a classificação *zero-shot* e *seen* numa únic framework.


[^1]: http://www.robotics.stanford.edu/~ang/papers/icml11-EncodingVsTraining.pdf
[^2]: https://nlp.stanford.edu/pubs/HuangACL12.pdf