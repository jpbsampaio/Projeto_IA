import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Rótulos

# Dividir o conjunto de dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo de árvore de decisão
clf = DecisionTreeClassifier()

# Treinar o modelo
clf.fit(X_train, y_train)

# Fazer previsões
y_pred = clf.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualizar a árvore de decisão
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names.tolist(), rounded=True)
plt.show()



# Explicação dos Passos
# Importações: Você importa as bibliotecas necessárias, incluindo numpy, scikit-learn para manipulação de dados e treinamento do modelo, e matplotlib para visualização.

# Carregar o Conjunto de Dados: Você carrega o conjunto de dados Iris e separa as características (X) dos rótulos (y).

# Dividir os Dados: Usa train_test_split para dividir os dados em conjuntos de treinamento (70%) e teste (30%).

# Criar e Treinar o Modelo: Cria um classificador de árvore de decisão (DecisionTreeClassifier) e treina o modelo com os dados de treinamento.

# Fazer Previsões: Usa o modelo treinado para fazer previsões no conjunto de teste.

# Avaliar o Modelo: Calcula a precisão das previsões comparando os rótulos previstos com os rótulos reais do conjunto de teste.

# Visualizar a Árvore de Decisão: Usa plot_tree do matplotlib para visualizar a árvore de decisão treinada.

# Resultado Esperado
# Precisão: O código imprimirá a precisão do modelo no console.
# Visualização: A visualização da árvore de decisão será mostrada em uma janela gráfica.
# Este é um exemplo completo e funcional de como usar uma árvore de decisão para classificação em IA. Se precisar de mais alguma coisa ou tiver alguma dúvida, é só avisar!