# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:20:18 2020

@author: Kiran
"""


import matplotlib.pyplot as plt
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import math
import sklearn.metrics as metrics

claim = pd.read_csv('C:\\Masters\\Semester1\\Machine Learing\\3.Assignment3\\SupportingDataSets\\claim_history.csv',
                       delimiter=',')
claim_train, claim_test = train_test_split(claim, test_size = 0.25, random_state = 60616, stratify = claim['CAR_USE'])

pd.set_option('display.max_columns',10)
pd.set_option('display.max_colwidth',100)
#  ----------------------------------------------------------------
class TreeNode:        
    def __init__(self,Predictorslist,Target, data,Entropy,level=1): 
        
        self.Predictorslist = Predictorslist
        self.Target = Target
        self.data = data
        self.Entropy = Entropy
        self.EventProb = class_Predictor(self.Target,self.data)
        
        self.Predictor = None  # Assign data 
        self.level = level
        self.LeftCriteria = None
        self.RightCriteria = None
        self.RightChild = None
        self.LeftChild = None
        
        if level <=2 :
            self.ConstructNode()
            if self.Entropy < Entropy :
                self.BuildPossibleChilds()            
        
        
    def listprint(self):
        printstring = ""
        for i in range(self.level-1):
            printstring+="\t"
            
        if self.LeftChild is None and self.RightChild is None:
            self.Entropy = cal_entropy(Target,self.data)
            print(printstring +  "Entropy: "+ str(self.Entropy))
            print(printstring +  "Number of Observations: "+ str(len(self.data)))
            for value in self.data[Target].unique():
                print(printstring+value + ":" + str(len(self.data[self.data[Target]==value])) )
            print(printstring +  "Commercial_Event_Prob: "+ str(class_Predictor(self.Target,self.data)))
        
        #if self.RightChild is not None or self.LeftChild is not None:
        elif self.Predictor is not None :
            print(printstring +  "Predictor: "+self.Predictor)
            if self.Entropy is not None:
                print(printstring +  "Entropy: "+ str(self.Entropy))
            if self.LeftCriteria is not None:
                print(printstring+ "LeftCriteria:"+ ",".join(self.LeftCriteria))        
            if self.LeftChild is not None:
                self.LeftChild.listprint()            
            if self.RightCriteria is not None:
                print(printstring+ "RightCriteria:"+",".join(self.RightCriteria))
            if self.RightChild is not None:
                self.RightChild.listprint()
        else :
            print(printstring + "This is a leaf node")
            
    def ConstructNode(self):
        Calculatedvalue = GetBestPredictor(Predictorslist,Target,self.data)
        if Calculatedvalue is not None and len(Calculatedvalue) > 0  :
            self.Predictor = (Calculatedvalue.Predictor)[0]
            self.Entropy = (Calculatedvalue.Entropy)[0]
            self.LeftCriteria = (Calculatedvalue.LeftBranch)[0]
            self.RightCriteria = (Calculatedvalue.RightBranch)[0]
        
    def BuildPossibleChilds(self):
        Leftfiltereddata = self.data[self.data[(self.Predictor)].isin((self.LeftCriteria))]
        leftchild = TreeNode(self.Predictorslist,self.Target,Leftfiltereddata,self.Entropy,self.level +1)
        
        Rightfiltereddata = self.data[self.data[(self.Predictor)].isin((self.RightCriteria))]
        rightchild = TreeNode(self.Predictorslist,self.Target,Rightfiltereddata,self.Entropy,self.level +1)  
            
        self.AddLeftChild(leftchild)        
        self.AddRightChild(rightchild)
        
    def AddLeftChild(self,node):
        self.LeftChild = node
        
    def AddRightChild(self,node):
        self.RightChild = node



            
#------------------
         
#------------------------------Functions--------------------------------------    
def cal_entropy(Target,dataset):
    temp = list(dataset.groupby(Target).size())
    temp = list(map(lambda x: (x/len(dataset))*math.log2(len(dataset)/x)*(-1),temp))
    return sum(temp)


def GetCombinations(datacolumn):    
    ocu_set = list(datacolumn.unique())
    combinationSet = []
    for i in range(1,math.ceil((len(ocu_set)+1)/2)):
        combinations = (list(itertools.combinations(ocu_set,i)))
        combinations = list(map(lambda x: list(x),combinations))
        combinationSet.extend(combinations)
    return combinationSet

def class_Predictor(Target,dataset):
    temp = (dataset.groupby(Target).size())    
    tempdf = pd.DataFrame(temp).reset_index()
    tempdf.columns = [Target,'Prob']   
    tempdf['Prob'] = tempdf['Prob'].apply(lambda x: x/sum(tempdf.Prob))
    tempclass = tempdf[tempdf[Target]=='Commercial']['Prob'][0]
    return tempclass



def CalulateEntropyForcombinations(predictor,Target,dataset):    
    datacolumn = dataset[predictor]
    CombinationSet = GetCombinations(datacolumn)
    
    SplitEntropyDataframe = pd.DataFrame(data={'Predictor':[],'LeftBranch':[],'LeftBranchEntropy':[],'RightBranch':[],'RightBranchEntropy':[],'Entropy':[]})
    
    for combination in CombinationSet:
        Split1Data = dataset[dataset[predictor].isin(combination)]
        Split2Data = dataset[~dataset[predictor].isin(combination)]          
        Split1DataEntropy = cal_entropy(Target,Split1Data)
        Split2DataEntropy = cal_entropy(Target,Split2Data)        
        WeightedEntropy = (len(Split1Data)/ len(dataset)) * Split1DataEntropy + (len(Split2Data)/ len(dataset)) * Split2DataEntropy        
        SplitEntropyDataframe = SplitEntropyDataframe.append({'Predictor':predictor,'LeftBranch':(combination),'LeftBranchEntropy':Split1DataEntropy,'RightBranch':list(set(dataset[predictor].unique())-set(combination)),'RightBranchEntropy':Split2DataEntropy,'Entropy':WeightedEntropy},ignore_index=True)
        
    return SplitEntropyDataframe

def GetBestPredictor(predictorslist,Target,dataset):
    BestPredictor = None
    for predictor in predictorslist:
        PredictorEntropy = CalulateEntropyForcombinations(predictor,Target,dataset)
        if len(PredictorEntropy) >0:
            MinEntropy = PredictorEntropy.iloc[[PredictorEntropy.Entropy.idxmin()]]        
            if BestPredictor is None and len(MinEntropy)> 0:
                BestPredictor = MinEntropy.reset_index(drop=True)
            elif len(MinEntropy)>0 and list(MinEntropy.Entropy)[0]< list(BestPredictor.Entropy)[0]:
                BestPredictor = MinEntropy.reset_index(drop=True)
    return BestPredictor


def CalculateAccuracy(predProbY,Y,Threshold):
    nY = len(Y)
    # Determine the predicted class of Y
    predY = numpy.empty_like(Y)
    
    for i in range(nY):
        if (predProbY[i] > Threshold):
            predY[i] = 'Commercial'
        else:
            predY[i] = 'Private'
    # Calculate the Root Average Squared Error
    RASE = 0.0
    for i in range(nY):
        if (Y[i] == 'Commercial'):
            RASE += (1 - predProbY[i])**2
        else:
            RASE += (0 - predProbY[i])**2
    RASE = numpy.sqrt(RASE/nY)
    
    # Calculate the Root Mean Squared Error
    Y_true = 1.0 * numpy.isin(Y, ['Commercial'])
    RMSE = metrics.mean_squared_error(Y_true, predProbY)
    RMSE = numpy.sqrt(RMSE)
    
    # For binary y_true, y_score is supposed to be the score of the class with greater label.
    AUC = metrics.roc_auc_score(Y_true, predProbY)
    accuracy = metrics.accuracy_score(Y, predY)
    
    print('                  Accuracy: {:.13f}' .format(accuracy))
    print('    Misclassification Rate: {:.13f}' .format(1-accuracy))
    print('          Area Under Curve: {:.13f}' .format(AUC))
    print('Root Average Squared Error: {:.13f}' .format(RASE))
    print('   Root Mean Squared Error: {:.13f}' .format(RMSE))    
    return


#-----------------------------------------------------------        

Predictorslist = ['OCCUPATION','CAR_TYPE','EDUCATION']
Target = 'CAR_USE'
maxlevel = 2

# Entropy of Root Node
root_ent =  cal_entropy(Target,claim_train)
print(root_ent)
  

DecisionTree = (TreeNode(Predictorslist,Target,claim_train,root_ent))

DecisionTree.listprint()

Root_Left_Left_data = DecisionTree.LeftChild.LeftChild.data
Root_Left_Left_data['Pred_Prob'] = DecisionTree.LeftChild.LeftChild.EventProb

Root_Left_Right_data = DecisionTree.LeftChild.RightChild.data
Root_Left_Right_data['Pred_Prob'] = DecisionTree.LeftChild.RightChild.EventProb

Root_Right_Left_data = DecisionTree.RightChild.LeftChild.data
Root_Right_Left_data['Pred_Prob'] = DecisionTree.RightChild.LeftChild.EventProb

Root_Right_Right_data = DecisionTree.RightChild.RightChild.data
Root_Right_Right_data['Pred_Prob'] = DecisionTree.RightChild.RightChild.EventProb


train_data = pd.concat([Root_Left_Left_data,Root_Left_Right_data,Root_Right_Left_data,Root_Right_Right_data])

fpr, tpr, thresholds = metrics.roc_curve(train_data[Target], train_data['Pred_Prob'], pos_label = 'Commercial')

# Kolmogorov Smirnov curve
cutoff = numpy.where(thresholds > 1.0, numpy.nan, thresholds)
plt.plot(cutoff, tpr, marker = 'o', label = 'True Positive',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(cutoff, fpr, marker = 'o', label = 'False Positive',
         color = 'red', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
print("***********************************")
print("Q2 (f)")
print("Cut off: ",thresholds)
print("KS statistic", tpr-fpr)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True, fontsize = 'large')
plt.show()


print("From Plot, we see highest difference is at:"+ "0.53419726")

#------------------------Question 3.....................




left_data = claim_test[claim_test.OCCUPATION.isin(DecisionTree.LeftCriteria)]

right_data = claim_test[claim_test.OCCUPATION.isin(DecisionTree.RightCriteria)]

ldf1=left_data[left_data.EDUCATION.isin(DecisionTree.LeftChild.LeftCriteria)]
ldf2=left_data[left_data.EDUCATION.isin(DecisionTree.LeftChild.RightCriteria)]

rdf1=right_data[right_data.CAR_TYPE.isin(DecisionTree.RightChild.LeftCriteria)]
rdf2=right_data[right_data.CAR_TYPE.isin(DecisionTree.RightChild.RightCriteria)]

t1= ldf1.groupby('CAR_USE').size()
ldf1['PRED_PROB_COM'] =  t1[0]/(t1[0]+t1[1])

t2= ldf2.groupby('CAR_USE').size()
ldf2['PRED_PROB_COM'] =  t2[0]/(t2[0]+t2[1])

t3= rdf1.groupby('CAR_USE').size()
rdf1['PRED_PROB_COM'] =  t3[0]/(t3[0]+t3[1])

t4= rdf2.groupby('CAR_USE').size()
rdf2['PRED_PROB_COM'] =  t4[0]/(t4[0]+t4[1])

test_data = pd.concat([ldf1,ldf2,rdf1,rdf2])




Y = list(test_data['CAR_USE'])

predProbY = list(test_data['PRED_PROB_COM'])

Training_Commercial_Proportion = len(claim_train[claim_train[Target]=='Commercial'])/len(claim_train)

CalculateAccuracy(predProbY,Y,Training_Commercial_Proportion)


# calculate accuracy with KS thresold -> 0.53419726

CalculateAccuracy(predProbY,Y,0.53419726)

#gini index

Y = test_data['CAR_USE'].to_numpy()
predProbY = test_data['PRED_PROB_COM'].to_numpy()

target_and_predictedprob=numpy.concatenate((Y.reshape(len(Y),1), predProbY.reshape(len(predProbY),1)), axis=1)

com=target_and_predictedprob[target_and_predictedprob[:,0] == 'Commercial']
com[:,1]=numpy.sort(com[:,1])
pvt=target_and_predictedprob[target_and_predictedprob[:,0] == 'Private']
pvt[:,1]=numpy.sort(pvt[:,1])


con=0
dis=0
tie=0

for i in com[:,1]:
    for j in pvt[:,1]:
        if i>j:
            con+=1
        elif i==j:
            tie+=1
        else:
            dis+=1

            
print("Gini Coeffiecient:")  
pairs=con+dis+tie
print((con-dis)/pairs)

print("Goodman-Kruskal Gamma statistic :")
print((con-dis)/(con+dis))



# Generate the coordinates for the ROC curve
OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(Y, predProbY, pos_label = 'Commercial')

# Add two dummy coordinates
OneMinusSpecificity = numpy.append([0], OneMinusSpecificity)
Sensitivity = numpy.append([0], Sensitivity)

OneMinusSpecificity = numpy.append(OneMinusSpecificity, [1])
Sensitivity = numpy.append(Sensitivity, [1])

# Draw the ROC curve
plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

