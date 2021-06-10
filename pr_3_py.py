import sklearn #Импорт Scikit-Learn

from sklearn.datasets import load_breast_cancer #Импорт набора данных
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score 
data = load_breast_cancer() 

label_names = data['target_names'] #Имена меток классификации 
labels = data['target'] #Фактические метки
feature_names = data['feature_names'] #Имена атрибутов / функций 
features = data['data'] #Атрибут

print(label_names, "\n") # печатаем метки классов
print(labels[0], "\n") 

#создадуние имен и значений функций
print(feature_names[0], "\n")
print(features[0], "\n")

train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.40, random_state = 42) #Организация данных в наборы

gnb = GaussianNB() 
model = gnb.fit(train, train_labels) 

preds = gnb.predict(test)
print(preds, "\n")
print(accuracy_score(test_labels, preds), "\n")
 
preds = gnb.predict(train)
print(preds, "\n")
print(accuracy_score(train_labels, preds), "\n")
