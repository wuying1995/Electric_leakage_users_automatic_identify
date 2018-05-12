
import pandas as pd
from random import shuffle
from sklearn.tree  import DecisionTreeClassifier  # 导入决策树模型

from sklearn.externals import joblib  #保存模型
from sklearn.metrics import confusion_matrix


datafile = 'Taxevasion identification.xls'


treefile='../tree2.pkl' #模型输出名字
tree = DecisionTreeClassifier()#建立决策树模型
#tree.fit(train[:,:3],train[,3]) #训练


joblib.dump(tree,treefile)


#cm_plot(train[:3],tree.predict(train[:,:3])).show()
