from sklearn.model_selection import GridSearchCV
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, make_scorer

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# import numpy as np
# from catboost imports CatBoostRegressor
# def model_score(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     score = (tp + tn) / (tp + fp + fn + tn)  # 计算准确率
#     return score


def NB_predict(X, y, q):
    classifier = MultinomialNB()
    classifier.fit(X, y)
    y_pred = classifier.predict(q)  # 預測

    return y_pred


def NB_modle(X, y):

    # 以下將自身資料切成train及test兩組，重新訓練一次，測試模型準確率

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30)  # 隨機挑選30%當測試資料
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)  # 訓練

    y_pred = classifier.predict(X_test)  # 用測試資料預測
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))  # 比對答案，計算準確率
    print(classification_report(y_test, y_pred))  # 印出分類報告
    print(confusion_matrix(y_test, y_pred, labels=['看漲', '看跌']))  # 印出混淆矩陣


def DecisionTree_modle(X, y):
    classifier = DecisionTreeClassifier(criterion="entropy")
    scores = cross_val_score(classifier, X, y, cv=5,
                             scoring='accuracy')  # 交叉驗證，計算準確率
    y_pred = cross_val_predict(classifier, X, y, cv=5)
    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)

    print(scores)
    print("Avg. Accuracy:", scores.mean())


def SVC_modle(X, y):

    classifier = SVC(kernel='linear')

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(classifier, X, y, cv=5,
                             scoring='accuracy')  # 交叉驗證，計算準確率
    y_pred = cross_val_predict(classifier, X, y, cv=5)
    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)
    print(scores)
    print("Avg. Accuracy:", scores.mean())


def XGboost(X, y):
    classifier = XGBClassifier()
    scores = cross_val_score(
        classifier, X, y, cv=5, error_score='raise', scoring='accuracy')
    y_pred = cross_val_predict(classifier, X, y, cv=5)
    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)
    print("Cross-validation scores:", scores)
    print("Mean accuracy:", scores.mean())


def RF_model(X, y):
    classifier = RandomForestClassifier(
        n_estimators=100, max_features='sqrt', random_state=42)
    y_pred = cross_val_predict(classifier, X, y, cv=5)
    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)
    scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
    mean_accuracy = scores.mean()
    print("Mean accuracy:", mean_accuracy)


def GBC_model(X, y):
    classifier = GradientBoostingClassifier(
        learning_rate=0.005, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30)
    param_grid = {'n_estimators': [50, 100, 150, 200, 250]}
    grid_search = GridSearchCV(
        classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = cross_val_predict(classifier, X, y, cv=5)
    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)
    scores = cross_val_score(best_model, Ｘ, y, cv=5)
    print("Best number of estimators:", best_model.n_estimators)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
