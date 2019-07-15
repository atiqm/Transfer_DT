#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 03:40:51 2019

@author: mounir
"""
import os,sys
sys.path.insert(0,'..')
sys.path.insert(0,'../Class_Imb_Ser/')
import lib_tree
import ser 

# =============================================================================
# 
# =============================================================================

import numpy as np
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_I():
    from sklearn.datasets import load_iris
    iris = load_iris()
    
    inds = np.where(iris.data[:,3] > np.median(iris.data[:,3]))[0]
    indt = np.where(iris.data[:,3] <= np.median(iris.data[:,3]))[0]
    
    X_source = iris.data[np.concatenate((inds,indt[:5]))]
    y_source = iris.target[np.concatenate((inds,indt[:5]))]


    X_target_005 = iris.data[np.concatenate((inds[-10:],indt[:5]))][::2]
    y_target_005 = iris.target[np.concatenate((inds[-10:],indt[:5]))][::2]

    X_target_095 = iris.data[np.concatenate((inds[-10:],indt[:5]))][1::2]
    y_target_095 = iris.target[np.concatenate((inds[-10:],indt[:5]))][1::2]
    return [X_source, X_target_005,
            X_target_095, y_source,
            y_target_005, y_target_095]
    
def load_6():
    from sklearn.datasets import load_digits
    digits = load_digits()
    
    X = digits.data[:200]
    y = (digits.target[:200] == 6).astype(int)
    
    X_targ = digits.data[200:]
    y_targ = (digits.target[200:] == 9 ).astype(int)
    
    X_source = X
    y_source = y
    
    # separating 5% & 95% of target data, stratified, random
    X_target_095, X_target_005, y_target_095, y_target_005 = train_test_split(
            X_targ,
            y_targ,
            test_size=0.05,
            stratify= y_targ)

    return [X_source, X_target_005,
            X_target_095, y_source,
            y_target_005, y_target_095]
    
# =============================================================================
# 
# =============================================================================


print('EXAMPLE SER')


#X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_I()
X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_6()
 
MAX = 5
solo_tree = False

def true_pos(clf,X,y):
    return sum(clf.predict(X[y==1]) == 1)/sum(y==1)
def false_pos(clf,X,y):
    return sum(clf.predict(X[y==0]) == 1)/sum(y==0)  

if solo_tree:

# =============================================================================
#     ON A UNIQUE DECISION TREE 
# =============================================================================
        
    dtree_or = DecisionTreeClassifier(max_depth=MAX)

    dtree_or.fit(X_source,y_source)

    dts = np.zeros(13,dtype=object)

    cl_no_red = [1]
    Nkmin = sum(y_target_005 == cl_no_red )
    root_source_values = lib_tree.get_node_distribution(dtree_or, 0).reshape(-1)

    props_s = root_source_values
    props_s = props_s / sum(props_s)
    props_t = np.zeros(props_s.size)
    for k in range(props_s.size):
        props_t[k] = np.sum(y_target_005 == k) / y_target_005.size
    
    coeffs = np.divide(props_t, props_s)        
    dts[1] = copy.deepcopy(dtree_or)
    ser.SER(0,dts[1], X_target_005, y_target_005, original_ser=True )    

    dts[2] = copy.deepcopy(dtree_or)
    ser.SER(0,dts[2], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True,cl_no_red= cl_no_red)
    
    dts[3] = copy.deepcopy(dtree_or)
    ser.SER(0,dts[3], X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext= cl_no_red )    

    #no ext vars
    dts[4] = copy.deepcopy(dtree_or)
    ser.SER(0,dts[4], X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext= cl_no_red ,ext_cond=True)
    dts[5] = copy.deepcopy(dtree_or)
    ser.SER(0,dts[5], X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext= cl_no_red ,leaf_loss_quantify=True, leaf_loss_threshold = 0.2,coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)    
    dts[6] = copy.deepcopy(dtree_or)
    ser.SER(0,dts[6], X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext= cl_no_red ,ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.9,coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)    
    
    #no red vars
    dts[7] = copy.deepcopy(dtree_or)
    ser.SER(0,dts[7], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=cl_no_red ,ext_cond=True)
    dts[8] = copy.deepcopy(dtree_or)
    ser.SER(0,dts[8], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red= cl_no_red ,leaf_loss_quantify=True, leaf_loss_threshold = 0.2,coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)    
    dts[9] = copy.deepcopy(dtree_or)
    ser.SER(0,dts[9], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red= cl_no_red ,ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.9,coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)    
       
    dts[10] = copy.deepcopy(dtree_or)
    ser.SER(0,dts[10], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red= cl_no_red , no_ext_on_cl=True, cl_no_ext= cl_no_red )
    dts[11] = copy.deepcopy(dtree_or)
    ser.SER(0,dts[11], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red= cl_no_red ,no_ext_on_cl=True, cl_no_ext= cl_no_red ,ext_cond=True)  
    
    dts[12] = copy.deepcopy(dtree_or)
    ser.SER(0,dts[12], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red= cl_no_red ,no_ext_on_cl=True, cl_no_ext= cl_no_red ,ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.5,coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)    
    dts[0] = copy.deepcopy(dtree_or)
    ser.SER(0,dts[0], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red= cl_no_red ,no_ext_on_cl=True, cl_no_ext= cl_no_red ,ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.9,coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)    

    n_star = 12
    print('score ser:', dts[1].score(X_target_095,y_target_095))
    print('score ser no red:', dts[2].score(X_target_095,y_target_095))
    print('score ser no ext:', dts[3].score(X_target_095,y_target_095))
    print('score ser *:', dts[n_star].score(X_target_095,y_target_095))

    print('tpr ser:', true_pos(dts[1],X_target_095,y_target_095))
    print('tpr ser no red:', true_pos(dts[2],X_target_095,y_target_095))
    print('tpr ser no ext:', true_pos(dts[3],X_target_095,y_target_095))
    print('tpr ser *:', true_pos(dts[n_star],X_target_095,y_target_095))

    print('fpr ser:', false_pos(dts[1],X_target_095,y_target_095))
    print('fpr ser no red:', false_pos(dts[2],X_target_095,y_target_095))
    print('fpr ser no ext:', false_pos(dts[3],X_target_095,y_target_095))
    print('fpr ser *:', false_pos(dts[n_star],X_target_095,y_target_095))
    
#    print('nb feuilles ser :',sum(dts[1].tree_.feature == -2))
#    print('nb feuilles ser no red :',sum(dts[2].tree_.feature == -2))
    
    
else:
    
# =============================================================================
#     ON A RANDOM FOREST
# =============================================================================
    
    N_EST = 3

    rf_or = RandomForestClassifier(n_estimators = N_EST,max_depth=MAX )
    rf_or.fit(X_source,y_source)

    rfs = np.zeros(13,dtype=object)
    
    rfs[1] = ser.SER_RF(rf_or, X_target_005, y_target_005, original_ser=True )    

    rfs[2] = ser.SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True,cl_no_red=[1] )
    rfs[3] = ser.SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext=[1] )    

    #no ext vars
    rfs[4] = ser.SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext=[1],ext_cond=True)
    rfs[5] = ser.SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext=[1],leaf_loss_quantify=True, leaf_loss_threshold = 0.2)    
    rfs[6] = ser.SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext=[1],ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.9)    
    
    #no red vars
    rfs[7] = ser.SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1],ext_cond=True)
    rfs[8] = ser.SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1],leaf_loss_quantify=True, leaf_loss_threshold = 0.2)    
    rfs[9] = ser.SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1],ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.9)    
       
    rfs[10] = ser.SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1], no_ext_on_cl=True, cl_no_ext=[1])
    rfs[11] = ser.SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1],no_ext_on_cl=True, cl_no_ext=[1],ext_cond=True)  
    
    rfs[12] = ser.SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1],no_ext_on_cl=True, cl_no_ext=[1],ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.1)    
    rfs[0] = ser.SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1],no_ext_on_cl=True, cl_no_ext=[1],ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.9)    

    n_star = 12
    dt = rfs[1].estimators_[0]
    dt_no_red = rfs[n_star].estimators_[0]
    
    print('score ser:', rfs[1].score(X_target_095,y_target_095))
    print('score ser no red:', rfs[2].score(X_target_095,y_target_095))
    print('score ser no ext:', rfs[3].score(X_target_095,y_target_095))
    print('score ser *:', rfs[n_star].score(X_target_095,y_target_095))

    print('tpr ser:', true_pos(rfs[1],X_target_095,y_target_095))
    print('tpr ser no red:', true_pos(rfs[2],X_target_095,y_target_095))
    print('tpr ser no ext:', true_pos(rfs[3],X_target_095,y_target_095))
    print('tpr ser *:', true_pos(rfs[n_star],X_target_095,y_target_095))

    print('fpr ser:', false_pos(rfs[1],X_target_095,y_target_095))
    print('fpr ser no red:', false_pos(rfs[2],X_target_095,y_target_095))
    print('fpr ser no ext:', false_pos(rfs[3],X_target_095,y_target_095))
    print('fpr ser *:', false_pos(rfs[n_star],X_target_095,y_target_095))
#    
#    print('nb feuilles ser :',sum(dt.tree_.feature == -2))
#    print('nb feuilles ser *:',sum(dt_no_red.tree_.feature == -2))

