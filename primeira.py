import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Treino: %i' % label)
    
   
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#Tutorial disponível online
clf1 = svm.SVC(gamma=0.001)
clf1.fit(data[:n_samples//2], digits.target[:n_samples//2])
# Teste svm
expected0 = digits.target[n_samples//2:]
predicted0 = clf1.predict(data[n_samples//2:])


#Algoritmo 1
clf2 = KNeighborsClassifier(n_neighbors = 9)
print(digits['target_names'])
#Dividindo a amostra em treino (70%) e teste(30%)
train, test, train_labels, test_labels = train_test_split(digits['data'], digits['target'], test_size=0.30, random_state=0)
clf2.fit(train, train_labels)

predicted1 = clf2.predict(test)

#Algoritmo 2
clf3 = DecisionTreeClassifier(max_depth=5)   
                                                  
#Dividindo a amostra em treino (70%) e teste(30%)
train2, test2, train_labels2, test_labels2 = train_test_split(digits['data'], digits['target'], test_size=0.30, random_state=0)
clf3.fit(train2, train_labels2)

predicted2 = clf3.predict(test2)

print("Relatório de classificação para classificador SVC \n  %s:\n%s\n" % (clf1, metrics.classification_report(expected0, predicted0)))
print("Relatório de classificação para classificador KNeighbors \n  %s:\n%s\n" % (clf2, metrics.classification_report(test_labels, predicted1)))
print("Relatório de classificação para classificador Decision Tree \n  %s:\n%s\n" % (clf3, metrics.classification_report(test_labels2, predicted2)))



images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted0))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
print(digits['DESCR'])