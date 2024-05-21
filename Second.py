import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns

# 1. Carregar o conjunto de dados a partir do arquivo local
file_path = "spambase.data"  # Certifique-se de que o arquivo está no mesmo diretório do script
columns = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
    "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order",
    "word_freq_mail", "word_freq_receive", "word_freq_will", "word_freq_people",
    "word_freq_report", "word_freq_addresses", "word_freq_free", "word_freq_business",
    "word_freq_email", "word_freq_you", "word_freq_credit", "word_freq_your", "word_freq_font",
    "word_freq_000", "word_freq_money", "word_freq_hp", "word_freq_hpl", "word_freq_george",
    "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology", "word_freq_1999",
    "word_freq_parts", "word_freq_pm", "word_freq_direct", "word_freq_cs", "word_freq_meeting",
    "word_freq_original", "word_freq_project", "word_freq_re", "word_freq_edu", "word_freq_table",
    "word_freq_conference", "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$",
    "char_freq_#", "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total", "label"
]

data = pd.read_csv(file_path, header=None, names=columns)

# 2. Preparar os dados
X = data.iloc[:, :-1]  # Todas as colunas menos a última são características
y = data.iloc[:, -1]   # A última coluna é o rótulo

# 3. Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Criar e treinar o modelo de árvore de decisão
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 5. Fazer previsões
y_pred = clf.predict(X_test)

# 6. Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Relatório de classificação
print(classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"]))

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Spam", "Spam"], yticklabels=["Not Spam", "Spam"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualizar a Árvore de Decisão
plt.figure(figsize=(20,10))  # Ajuste o tamanho da figura para melhor visualização
plot_tree(clf, filled=True, feature_names=columns[:-1], class_names=["Not Spam", "Spam"], rounded=True, fontsize=8)
plt.show()

#visualizar o grafico de pizza
labels = ["Not Spam", "Spam"]
sizes = [sum(y_test == 0), sum(y_test == 1)]
colors = ['skyblue', 'lightcoral']
explode = (0.1, 0)  # explode 1st slice (Not Spam)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Distribution of Not Spam and Spam in Test Set')
plt.show()