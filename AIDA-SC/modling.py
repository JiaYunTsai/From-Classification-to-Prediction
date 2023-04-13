from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix  

from sklearn.tree import DecisionTreeClassifier     
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

def NB_predict(X, y, q):
    classifier = MultinomialNB() 
    classifier.fit(X, y)
    y_pred = classifier.predict(q) #預測

    return y_pred

def NB_modle(X, y):

    #以下將自身資料切成train及test兩組，重新訓練一次，測試模型準確率

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) #隨機挑選30%當測試資料
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train) #訓練

    y_pred= classifier.predict(X_test) #用測試資料預測
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #比對答案，計算準確率
    print(classification_report(y_test, y_pred)) #印出分類報告
    print(confusion_matrix(y_test, y_pred, labels=['看漲','看跌'])) #印出混淆矩陣

def DecisionTree_modle(X, y):
    classifier = DecisionTreeClassifier(criterion="entropy")
    scores = cross_val_score(classifier,X,y,cv=5,scoring='accuracy') #交叉驗證，計算準確率
    print(scores)
    print("Avg. Accuracy:",scores.mean())

def SVC_modle(X, y):

    classifier = SVC(kernel='linear')

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(classifier,X,y,cv=5,scoring='accuracy') #交叉驗證，計算準確率
    print(scores)
    print("Avg. Accuracy:",scores.mean())