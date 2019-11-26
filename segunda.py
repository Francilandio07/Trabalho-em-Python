from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

#Dividindo a amostra com 20% dos dados para teste e 80% para treino
train, test, train_labels, test_labels = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=0)

#Teste com KNeighbors
clf1 = KNeighborsClassifier(n_neighbors=5)  
clf1.fit(train, train_labels)
predicted = clf1.predict(test)

print("Relatório de classificação para classificador KNeighbors \n  %s:\n%s\n" % (clf1, metrics.classification_report(test_labels, predicted)))

#Teste com SVC
clf2 = SVC(gamma ='scale')
clf2.fit(train, train_labels)
predicted1 = clf2.predict(test)
print("Relatório de classificação para classificador SVC \n  %s:\n%s\n" % (clf2, metrics.classification_report(test_labels, predicted1)))

#Teste com Decision Tree
clf3 = DecisionTreeClassifier(max_depth=5)
clf3.fit(train, train_labels)
predicted2 = clf3.predict(test)
print("Relatório de classificação para classificador Decision Tree \n  %s:\n%s\n" % (clf3, metrics.classification_report(test_labels, predicted2)))

#Teste com GaussianNB
clf4 = GaussianNB()
clf4.fit(train, train_labels)
predicted3 = clf4.predict(test)
print("Relatório de classificação para classificador GaussianNB \n  %s:\n%s\n" % (clf4, metrics.classification_report(test_labels, predicted3)))
print(iris['DESCR'])