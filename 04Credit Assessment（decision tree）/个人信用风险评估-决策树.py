# 导入相关库包# 导入所需的库和模块
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from six import StringIO
from IPython.display import Image
import pydot
from sklearn import tree
sns.set()

# 载入数据,查看贷款违规行为以及用户的借贷、储蓄情况
credit = pd.read_csv('credit.csv')
print('Data size: ', credit.shape)
print('Bad samples:', (credit['default'] - 1).sum())
print('Good samples:', credit.shape[0] - (credit['default'] - 1).sum())
print(credit.checking_balance.value_counts())
print(credit.savings_balance.value_counts())

col_dicts = {}
cols = ['checking_balance', 'credit_history', 'purpose', 'savings_balance', 'employment_length', 'personal_status',
        'other_debtors', 'property', 'installment_plan', 'housing', 'job', 'telephone', 'foreign_worker']
col_dicts = {'checking_balance': {'1 - 200 DM': 2,
                                  '< 0 DM': 1,
                                  '> 200 DM': 3,
                                  'unknown': 0},
             'credit_history': {'critical': 0,
                                'delayed': 2,
                                'fully repaid': 3,
                                'fully repaid this bank': 4,
                                'repaid': 1},
             'employment_length': {'0 - 1 yrs': 1,
                                   '1 - 4 yrs': 2,
                                   '4 - 7 yrs': 3,
                                   '> 7 yrs': 4,
                                   'unemployed': 0},
             'foreign_worker': {'no': 1, 'yes': 0},
             'housing': {'for free': 1, 'own': 0, 'rent': 2},
             'installment_plan': {'bank': 1, 'none': 0, 'stores': 2},
             'job': {'mangement self-employed': 3,
                     'skilled employee': 2,
                     'unemployed non-resident': 0,
                     'unskilled resident': 1},
             'other_debtors': {'co-applicant': 2, 'guarantor': 1, 'none': 0},
             'personal_status': {'divorced male': 2,
                                 'female': 1,
                                 'married male': 3,
                                 'single male': 0},
             'property': {'building society savings': 1,
                          'other': 3,
                          'real estate': 0,
                          'unknown/none': 2},
             'purpose': {'business': 5,
                         'car (new)': 3,
                         'car (used)': 4,
                         'domestic appliances': 6,
                         'education': 1,
                         'furniture': 2,
                         'others': 8,
                         'radio/tv': 0,
                         'repairs': 7,
                         'retraining': 9},
             'savings_balance': {'101 - 500 DM': 2,
                                 '501 - 1000 DM': 3,
                                 '< 100 DM': 1,
                                 '> 1000 DM': 4,
                                 'unknown': 0},
             'telephone': {'none': 1, 'yes': 0}}
data=credit.copy()
for col in cols:
    data[col] = data[col].map(col_dicts[col])
print(data.head(10))

# 准备特征和目标变量
X = data.drop('default', axis=1)  # 特征矩阵
y = data['default']  # 目标变量
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(y_train.value_counts()/len(y_train))
print(y_test.value_counts()/len(y_test))

# 创建决策树分类器
credit_model = DecisionTreeClassifier(criterion = 'entropy') #使用信息熵
#credit_model = DecisionTreeClassifier(criterion = 'gini') #Gini系数
# 在训练集上训练模型
credit_model.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = credit_model.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("模型在测试集上的准确率: {:.2f}".format(accuracy))
credit_pred = credit_model.predict(X_test)

from sklearn import metrics
print( metrics.classification_report(y_test, credit_pred))
print(metrics.confusion_matrix(y_test, credit_pred))
print(metrics.accuracy_score(y_test, credit_pred))

# #将决策树以图片的形式输出
# dot_data = StringIO()
# tree.export_graphviz(credit_model, out_file = dot_data,
#                          feature_names = X_train.columns,
#                          class_names=['no default','default'],
#                          filled = True, rounded = True,
#                          special_characters = True)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# # 访问第一个图形对象（假设只有一个图形对象）
# graph_image = graph[0].create_png()

# # 保存为PNG文件
# with open('decision_tree.png', 'wb') as f:
#     f.write(graph_image)

# # 显示图像
# Image(graph_image)
