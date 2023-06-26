from sklearn.linear_model import LogisticRegression # ロジスティクス回帰
from sklearn import tree                            # 決定木
from sklearn.ensemble import RandomForestClassifier # ランダムフォーレスト
from sklearn.svm import SVC                         # サポートベクトルマシーン


# 万能関数
def my_model(X_train, X_test, y_train, y_test, model, name=''):
  print('モデル名:{}'.format(name))
  model.fit(X_train, y_train)
  pred = model.predict(X_test)
  my_evaluation(y_test, pred)


from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

def my_evaluation(y, pred):
  # 正解と予測値を入れると自動で評価してくれます。
  print('混同行列:')
  print(confusion_matrix(y, pred))
  accuracy = accuracy_score(y, pred)
  precision = precision_score(y, pred)
  recall = recall_score(y, pred)
  f1 = f1_score(y, pred)

  print("正解率: %.3f" % accuracy)
  print("精度: %.3f" % precision)
  print("再現率: %.3f" % recall)
  print("F1スコア: %.3f" % f1)


def help():
  print('関数1: my_model')
  print("my_model(X_train, X_test, y_train, y_test, model, name='')\n")
  print('関数2: my_evaluation')
  print('my_evaluation(y, pred)\n')
  print('読み込み済みのモデル:')
  print('tree, LogisticRegression, RandomForestClassifier, SVC')