# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 20:08:42 2016

@author: Levy
"""
# 载入所需要的库
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# 载入学生数据集
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"

student_data=pd.DataFrame(student_data) 
# TODO： 计算学生的数量
print len(student_data)
n_students = len(student_data)

# TODO： 计算特征数量
n_features = len(student_data.columns)

# TODO： 计算通过的学生数
n_passed = len(student_data['passed'][student_data['passed']=='yes'])

# TODO： 计算未通过的学生数
n_failed = len(student_data['passed'][student_data['passed']=='no'])

# TODO： 计算通过率
grad_rate = 100.0*n_passed/n_students

# 输出结果
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# 提取特征列
feature_cols = list(student_data.columns[:-1])

# 提取目标列 ‘passed’
target_col = student_data.columns[-1] 

# 显示列的列表
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# 将数据分割成特征数据和目标数据（即X_all 和 y_all）
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# 通过打印前5行显示特征信息
#print "\nFeature values:"
#print X_all.head()

def preprocess_features(X):
    ''' 预处理学生数据，将非数字的二元特征转化成二元值（0或1），将分类的变量转换成虚拟变量
    '''
    
    # 初始化一个用于输出的DataFrame
    output = pd.DataFrame(index = X.index)

    # 查看数据的每一个特征列
    for col, col_data in X.iteritems():
        
        # 如果数据是非数字类型，将所有的yes/no替换成1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # 如果数据类型是类别的（categorical），将它转换成虚拟变量
        if col_data.dtype == object:
            # 例子: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # 收集转换后的列
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)

#print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
X_all = pd.DataFrame(X_all) 

y_all = pd.Series(y_all)
x= X_all.values
#x=np.array((X_all.as_matrix()))
y=[[i] for i in y_all]
#column_or_1d(y, warn=True)
#print x
# TODO：在这里导入你可能需要使用的另外的功能
from sklearn.cross_validation import train_test_split

#random_data = np.hstack((x,y))
#map(int ,random_data)
# TODO：设置训练集的数量
num_train = 300

# TODO：设置测试集的数量
num_test = X_all.shape[0] - num_train


# TODO：把数据集混洗和分割成上面定义的训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X_all,y_all,test_size=num_test,
                                               train_size=num_train,random_state=1)
#shuffle(random_data,random_state=10)
#X_train = random_data[:, :-1][:num_train]
#map(map,[float for i in range(len(X_train))],X_train)
#X_train=np.array(X_train,float)
#X_train=pd.DataFrame(X_train)
#X_train=np.array(X_train,int)
#X_test = random_data[:, :-1][num_train:]
#X_test=pd.DataFrame(X_test)
#X_test=np.array(X_test,float)
#X_test=np.array(X_test,int)
#y_train = random_data[:, -1][:num_train]
#y_train = pd.Series(y_train)
#y_test = random_data[:, -1][num_train:]
#y_test = pd.Series(y_test)
# 显示分割的结果
#print "Training set has {} samples.".format(X_train.shape[0])
#print "Testing set has {} samples.".format(X_test.shape[0])

def train_classifier(clf, X_train, y_train):
    ''' 用训练集训练分类器 '''
    
    # 开始计时，训练分类器，然后停止计时
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' 用训练好的分类器做预测并输出F1值'''
    
    # 开始计时，作出预测，然后停止计时
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # 输出并返回结果
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' 用一个分类器训练和预测，并输出F1值 '''
    
    # 输出分类器名称和训练集大小
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # 训练一个分类器
    train_classifier(clf, X_train, y_train)
    
    # 输出训练和测试的预测结果
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))


    
# TODO：从sklearn中引入三个监督学习模型
#from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import svm
from sklearn import linear_model

# TODO：初始化三个模型
#clf_D = naive_bayes.GaussianNB()
clf_A = ensemble.AdaBoostClassifier(random_state=10)
clf_B = svm.SVC(random_state=10)
clf_C = linear_model.LogisticRegression(random_state=10)

# TODO：设置训练集大小
#y_train_num = np.array([])
#y_train_value = np.array([1 if i == 'yes' else 0 for i in y_train])
#y_test_value = np.array([1 if i == 'yes' else 0 for i in y_test])
#y_test_dict = {}
#X_train_df= pd.DataFrame(X_train)
#y_train_se= pd.Series(y_train)
#X_test_df = pd.DataFrame(X_test)
#y_test_se = pd.Series(y_test)

X_train_100 = X_train[:100]
y_train_100 = y_train[:100]

X_train_200 = X_train[:200]
y_train_200 = y_train[:200]

X_train_300 = X_train[:300]
y_train_300 = y_train[:300]
#print 'test dict'
#print y_test_dict
# TODO：对每一个分类器和每一个训练集大小运行'train_predict' 
# train_predict(clf, X_train, y_train, X_test, y_test)    
#clf_A.fit(X_train_100,y_train_100)
#y_pred = clf_A.predict(X_train_100)

#train_predict(clf_C,X_train_df,y_train_se,X_test_df,y_test_se)  
  
for clf in [clf_A, clf_B, clf_C]:
    for x,y in [[X_train_100,y_train_100], [X_train_200,y_train_200], [X_train_300,y_train_300]]:
        train_predict(clf, x, y, X_test, y_test)

#train_predict(clf_B,X_train_300,y_train_300,X_test,y_test_value)   
#train_predict(clf_C,X_train_300,y_train_300,X_test,y_test_value) 
#train_predict(clf_D,X_train_300,y_train_300,X_test,y_test_value) 
train_predict(clf_A, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_A, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_A, X_train_300, y_train_300, X_test, y_test)
train_predict(clf_B, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_B, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_B, X_train_300, y_train_300, X_test, y_test)
train_predict(clf_C, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_C, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_C, X_train_300, y_train_300, X_test, y_test)

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

parameters = {'kernel':['poly', 'rbf', 'sigmoid']
              ,'C':[i/20.0 for i in range(1,40)]
              }

clf=svm.SVC()

f1_scorer=make_scorer(f1_score,pos_label='yes')

grid_obj = GridSearchCV(clf,param_grid=parameters,scoring=f1_scorer)

grid_obj = grid_obj.fit(X_train,y_train)

default_param={'kernel':['rbf'],'C':[1.0]}

grid_default = GridSearchCV(clf,param_grid=default_param,scoring=f1_scorer)

grid_default=grid_default.fit(X_train,y_train)

print grid_default.best_score_
print grid_obj.best_params_
print grid_obj.best_score_
#print grid_obj.grid_scores_


clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
# 输出经过调参之后的训练集和测试集的F1值
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))


