import matplotlib.pyplot as plt
font = {'family': 'serif','color':  'green','weight': 'normal','size': 20}

#Parte 1
modelo1 = ['SVM', 'KNeighbors', 'DecisionTree']
acuracy1 = [97,98,72]
xs = [i + 0.5 for i, _ in enumerate(modelo1)]
margem_erro = [3, 2, 28]
plt.bar(xs, acuracy1, yerr= margem_erro)

plt.xlabel('Algoritmos utilizados', fontdict=font)
plt.ylabel('Acurácia (%)', fontdict=font)

# Ajuste do espaçamento para impedir o recorte de ylabel
plt.title('Parte 1', fontdict=font)
plt.xticks([i + 0.5 for i, _ in enumerate(modelo1)], modelo1)
plt.subplots_adjust(left=0.15)
plt.show()


#Parte 2
modelo2 = ['KNeighbors', 'SVC', 'DecisionTree', 'GaussianNB']
acuracy2 = [97, 100, 100, 97]
x = [j + 0.5 for j, _ in enumerate(modelo2)]
margem = [3, 0, 0, 3]
#Plotagem de gráfico de barras
plt.bar(x, acuracy2, yerr= margem)
#Títulos dos eixos
plt.xlabel('Algoritmos utilizados', fontdict=font)
plt.ylabel('Acurácia (%)', fontdict=font)

#Ajuste do espaçamento para impedir o recorte de ylabel 
plt.title('Parte 2', fontdict=font)
plt.xticks([j + 0.5 for j, _ in enumerate(modelo2)], modelo2)
plt.subplots_adjust(left=0.15)
plt.show()