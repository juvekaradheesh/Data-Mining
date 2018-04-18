import pandas as pd
import math
import numpy as np
from numpy import transpose
import matplotlib.pyplot as plt
from Tree import Node
from sklearn.preprocessing import LabelEncoder
import random


def calculateInfoD(label):
    classes = []
    # Create classes array with count
    for x in label:
        isfound = False
        for y in classes:
            if y[0] == x:
                isfound = True
                y[1] += 1
        if not isfound:
            classes.append([x,1])
    
    # Calculate the info(D)
    gain = 0
    total = len(label)
    for x in classes:
        gain += (x[1]/total)*math.log((total/x[1]),2)

    return gain

def calculateInfoA(attribute,label):
    classes = []
    # Create classes array with count for attribute
    for x in attribute:
        isfound = False
        for y in classes:
            if y[0] == x:
                isfound = True
                y[1] += 1
        if not isfound:
            classes.append([x,1])
    
    # Calculate the infoA(D)
    gain = 0
    total = len(attribute)
    for x in classes:
        tempLabel = []
        for i in range(len(label)):
            if attribute[i] == x[0]:
                tempLabel.append(label[i])
        gain += (x[1]/total)*calculateInfoD(tempLabel)
    return gain

def calculateGain(attribute,label,InfoD):
    return InfoD - calculateInfoA(attribute,label)


def getNode(dataset,deleteRows,label,InfoD,deleteCols):
    temp_dataset = dataset
    temp_dataset = np.delete(temp_dataset,deleteRows,1)
    temp_label = label
    temp_label = np.delete(temp_label,deleteRows)
    gainArray = []
    for i in range(len(temp_dataset)):
        if i not in deleteCols:
            gainArray.append(calculateGain(temp_dataset[i],temp_label,InfoD))
        else:
            gainArray.append(0)
    return gainArray.index(max(gainArray))    

def makeTree(dataset,deleteRows,label,InfoD,parentNode,edgeNum,deleteCols,dataRow):
    # node = getNode(dataset,deleteRows,label,InfoD,deleteCols)
    # Tree.add_node(node)
    # Tree.add_edge(parentNode,node,weight=edgeNum)
    
    nodeNum = getNode(dataset,deleteRows,label,InfoD,deleteCols)
    node = Node(nodeNum,parentNode)
    edge = {"edgeNum":edgeNum, "node":node}
    parentNode.genChildren(edge)
    parentNode = node

    elements = []
    for x in dataset[nodeNum]:
        if x not in elements:
            elements.append(x)
    deleteRows = []
    if nodeNum not in deleteCols:
        deleteCols.append(nodeNum)
    if len(deleteCols) is len(dataset):
        for element in elements:
            dataRow[nodeNum] = element
            flag = False
            store = 0
            for idx,x in enumerate(dataset.T):
                if np.array_equal(dataRow,x):
                    flag = True
                    store = idx
                    break
            if not flag:
                store = random.randint(0,len(dataset[0])-2)
            # store label[store] in node now
            temp_node = Node(-2,parentNode)
            edge = {"edgeNum":element, "node":temp_node, "answer":label[store]}
            parentNode.genChildren(edge)
            
        return 0

    for x in elements:
        dataRow[nodeNum] = x
        for i in range(len(dataset[nodeNum])):
            if x != dataset[nodeNum][i]:
                deleteRows.append(i)
        makeTree(dataset,deleteRows,label,InfoD,parentNode,x,deleteCols[:],dataRow[:])

def decisionTreeClassifier(dataset,label):
    InfoD = calculateInfoD(label)
    root = Node(-1,None)
    dataRow=[]
    for x in range(len(dataset)):
        dataRow.append(-1)
    makeTree(dataset,[],label,InfoD,root,0,[],dataRow[:])
    return root

def predict(dataset,root):
    predicted_values = []
    for x in range(len(dataset[0])):
        curr_node = root.children[0]['node']
        while curr_node.value is not -2:
            value = dataset[curr_node.value,x]
            for y in range(len(curr_node.children)):
                if curr_node.children[y]['edgeNum'] == value:
                    store = y
                    break
            curr_node = curr_node.children[store]['node']
        predicted_values.append(curr_node.parent.children[store]['answer'])
    return predicted_values


# df = pd.read_csv("data_temp.csv")
# root = decisionTreeClassifier(transpose(df.drop(['Class'],axis=1).as_matrix()),df['Class'].values.tolist())
# Y_predict = predict(transpose(df.drop(['Class'],axis=1).as_matrix()),root)
# print(Y_predict)



# df = pd.read_csv("processed_adult.csv")
# df_test = pd.read_csv("processed_test.csv")

# lb_make = LabelEncoder()
# df["Sex"] = lb_make.fit_transform(df["Sex"])
# df["Race"] = lb_make.fit_transform(df["Race"])
# df["Relationship"] = lb_make.fit_transform(df["Relationship"])
# df["Occupation"] = lb_make.fit_transform(df["Occupation"])
# df["Marital_stat"] = lb_make.fit_transform(df["Marital_stat"])
# df["WorkClass"] = lb_make.fit_transform(df["WorkClass"])
# df["Education"] = lb_make.fit_transform(df["Education"])
# df["Final_wt"] = lb_make.fit_transform(df["Final_wt"])
# df["Hr_per_week"] = lb_make.fit_transform(df["Hr_per_week"])

# df_test["Sex"] = lb_make.fit_transform(df_test["Sex"])
# df_test["Race"] = lb_make.fit_transform(df_test["Race"])
# df_test["Relationship"] = lb_make.fit_transform(df_test["Relationship"])
# df_test["Occupation"] = lb_make.fit_transform(df_test["Occupation"])
# df_test["Marital_stat"] = lb_make.fit_transform(df_test["Marital_stat"])
# df_test["WorkClass"] = lb_make.fit_transform(df_test["WorkClass"])
# df_test["Education"] = lb_make.fit_transform(df_test["Education"])
# df_test["Final_wt"] = lb_make.fit_transform(df_test["Final_wt"])
# df_test["Hr_per_week"] = lb_make.fit_transform(df_test["Hr_per_week"])

# df['Salary'] = lb_make.fit_transform(df['Salary'])
# df_test['Salary'] = lb_make.fit_transform(df_test['Salary'])

# X = df.drop(['Salary'],axis=1)
# Y = df['Salary'].values.tolist()

# X_test = df_test.drop(['Salary'],axis=1)
# Y_test = df_test['Salary'].values.tolist()

# X = transpose(X.as_matrix())

# X_test = transpose(X_test.as_matrix())

# root = decisionTreeClassifier(X,Y)
# print("Tree Created")
# Y_predicted = predict(X_test,root)
# print(Y_predicted)