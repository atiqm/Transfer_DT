#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:52:46 2019

@author: mounir
"""


import os,sys
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import copy


sys.path.insert(0,'../')
import lib_tree
# =============================================================================
#         
# =============================================================================

def threshold_selection(Q_source_parent,
                        Q_source_l,
                        Q_source_r,
                        X_target_node,
                        Y_target_node,
                        phi,
                        classes,
                        use_divergence=True,
                        measure_default_IG=True):
    # print("Q_source_parent : ", Q_source_parent)
    # sort the corrdinates of X along phi
    #X_phi_sorted = np.sort(X_target_node[:, phi])
    
    #modif pour ne considérer que les valeurs distinctes
    X_phi_sorted = np.array(list(set(X_target_node[:, phi])))
    X_phi_sorted = np.sort(X_phi_sorted)
    
    # print(X_phi_sorted)
    nb_tested_thresholds = X_phi_sorted.shape[0] - 1
    
    if nb_tested_thresholds == 0:
        return X_phi_sorted[0]
    
    measures_IG = np.zeros(nb_tested_thresholds)
    measures_DG = np.zeros(nb_tested_thresholds)
    for i in range(nb_tested_thresholds):
        threshold = (X_phi_sorted[i] + X_phi_sorted[i + 1]) * 1. / 2
        Q_target_l, Q_target_r = lib_tree.compute_Q_children_target(X_target_node,
                                                           Y_target_node,
                                                           phi,
                                                           threshold,
                                                           classes)

        measures_IG[i] = lib_tree.IG(Q_source_parent,
                            [Q_target_l, Q_target_r])
        measures_DG[i] = lib_tree.DG(Q_source_l,
                            Q_source_r,
                            Q_target_l,
                            Q_target_r)
    index = 0
    max_found = 0
    
    if use_divergence:
        for i in range(1, nb_tested_thresholds - 1):
            if measures_IG[i] >= measures_IG[i - 1] and measures_IG[i] >= measures_IG[i + 1] and measures_DG[i] > measures_DG[index]:
                max_found = 1
                index = i
                
        if not max_found :

            if measure_default_IG:
                index = np.argmax(measures_IG)
            else:
                index = np.argmax(measures_DG)
    #print('div index',index)
    else:    
        index = np.argmax(measures_IG)
    
    
    #print('no div index',np.argmax(measures_IG))
    #print('------')
    threshold = (X_phi_sorted[index] + X_phi_sorted[index + 1]) * 1. / 2
    return threshold


# =============================================================================
# 
# =============================================================================

def STRUT(decisiontree,
          node_index,
          X_target_node,
          Y_target_node,
          X_target_node_noupdate,
          Y_target_node_noupdate,
          pruning_updated_node=True,
          no_prune_on_cl=False,
          cl_no_prune=None,
          prune_lone_instance=True,
          adapt_prop=False,
          simple_weights=False,
          coeffs=[1, 1],
          use_divergence=True,
          measure_default_IG=True):

#    T = decisiontree.tree_
    phi = decisiontree.tree_.feature[node_index]
    classes = decisiontree.classes_
    threshold = decisiontree.tree_.threshold[node_index]
    old_threshold = threshold


    current_class_distribution_source = decisiontree.tree_.value[node_index].astype(int)
    #source_class_distribution = T.value[node_index].astype(int)
    current_class_distribution_noupdate = lib_tree.compute_class_distribution(classes, Y_target_node_noupdate)
    current_class_distribution = lib_tree.compute_class_distribution(classes, Y_target_node)

    decisiontree.tree_.weighted_n_node_samples[node_index] = Y_target_node.size
    decisiontree.tree_.impurity[node_index] = lib_tree.GINI(current_class_distribution)
    decisiontree.tree_.n_node_samples[node_index] = Y_target_node.size

    # If it is a leaf one, exit
    if decisiontree.tree_.children_left[node_index] == -2:
    #if tree.children_left[node_index] == -1 and tree.children_right[node_index] == -1:
        # print("it's a leaf")
        # update tree.value
        decisiontree.tree_.value[node_index] = current_class_distribution
        return node_index

    is_reached_update = (current_class_distribution.sum() != 0)
    is_reached_noupdate = (current_class_distribution_noupdate.sum() != 0)

    is_instance_cl_no_prune = np.sum(decisiontree.tree_.value[node_index, :,
                                                cl_no_prune].astype(int))
#     print("is_reached_update : ", is_reached_update)
#     print("is_reached_noupdate : ", is_reached_noupdate)

    # NEW prune_cond
    add_source_value = False
    # print("is_reached_update : ", is_reached_update)
    # print("is_reached_noupdate : ", is_reached_noupdate)
    # print("is_instance_cl_no_prune : ", is_instance_cl_no_prune)
    if pruning_updated_node:
        if no_prune_on_cl:
            # flag meaning need to add source value (to avoid zero !)
            add_source_value = not is_reached_update and is_instance_cl_no_prune
            prune_cond = not is_reached_update and not is_instance_cl_no_prune
        else:
            prune_cond = not is_reached_update
    else:
        if no_prune_on_cl:
            # flag meaning need to add source value (to avoid zero !)
            add_source_value = not is_reached_update and is_instance_cl_no_prune
            prune_cond = not is_reached_noupdate and not is_instance_cl_no_prune
        else:
            prune_cond = not is_reached_noupdate or not is_reached_update

    # OLD prune_cond
    # prune_cond = not is_reached_update or (not pruning_updated_node and (not is_reached_noupdate) and ((not no_prune_on_cl) or (not is_instance_cl_no_prune)))
    # if no target data at all or ((not reached) and (pruning activated or no
    # instance to preserve)), then prune

    if prune_cond:
        # print("PRUNING at node ", node_index)
#        prune_subtree(decisiontree,
#                      node_index)
#        parent_node, b_p = find_parent(tree, node_index)
#        # Get the brother index
#        if b_p == -1:  # current_node is left_children
#            brother_node = tree.children_right[parent_node]
#        if b_p == 1:  # current_node is right_children
#            brother_node = tree.children_left[parent_node]
#        # Get grand parent index
#        grand_parent_node, b_gp = find_parent(tree, parent_node)
#        # Shunt the parent
#        if b_gp == -1:  # parent is left_children of grandparent
#            tree.children_left[grand_parent_node] = brother_node
#
#        if b_gp == 1:  # parent is right_children of grandparent
#            tree.children_right[grand_parent_node] = brother_node
#        # supress the current node
#        # tree.children_left[node_index] = -1  # seem useless since already done in prune_subtree func
#        # tree.children_right[node_index] = -1
#        tree.children_left[parent_node] = -1  # important
#        tree.children_right[parent_node] = -1
        
        # =============================================================================
        #         
        # =============================================================================

        p,b = lib_tree.find_parent(decisiontree, node_index)
        node_index = lib_tree.cut_from_left_right(decisiontree,p,b)
        
        return node_index

    # else:
        # print("NO Pruning at node ", node_index)
        # return 0

    # update tree.value with target data
    decisiontree.tree_.value[node_index] = current_class_distribution
    if add_source_value:
        # print("adding source value to node")
        decisiontree.tree_.value[node_index] = current_class_distribution_source
        decisiontree.tree_.n_node_samples[node_index] = np.sum(current_class_distribution_source)
        decisiontree.tree_.weighted_n_node_samples[node_index] = np.sum(current_class_distribution_source)
        lib_tree.add_to_parents(decisiontree, node_index,current_class_distribution_source)
        
        return node_index

    # Only one class is present in the node -> terminal leaf
    if (current_class_distribution > 0).sum() == 1:
        # print("Only one class in node {} --> PRUNING".format(node_index))
        node_index = lib_tree.cut_into_leaf2(decisiontree, node_index)

        #prune_subtree(decisiontree,node_index)
        #tree.feature[node_index] = -2

        
        return node_index
    # Only one instance -> pruning into leaf
    # if current_class_distribution.sum() == 1:
        # # print("Only one instance in node {} --> PRUNING ? ".format(node_index))
        # if prune_lone_instance:
        # # print("YES")
        # prune_subtree(decisiontree,
        # node_index)
        # tree.feature[node_index] = -2
        # # else:
        # # if is_instance_cl_no_prune:
        # # print("NO")
        # return 0

    # update threshold
    if type(threshold) is np.float64:
        Q_source_l, Q_source_r = lib_tree.get_children_distributions(decisiontree,
                                                                node_index)
        Sl = np.sum(Q_source_l)
        Sr = np.sum(Q_source_r)


        if simple_weights:

            Q_source_l = np.multiply(coeffs, Q_source_l)
            Q_source_r = np.multiply(coeffs, Q_source_r)
            
#        if adapt_prop_hetero:
#            Q_source_l = np.multiply(coeffs, Q_source_l)
#            D = np.sum(Q_source_l)
#            Q_source_l = np.divide(Q_source_l,D)
#            Q_source_r = np.multiply(coeffs, Q_source_r)
#            D = np.sum(Q_source_r)
#            Q_source_r = np.divide(Q_source_r,D)  
            
        if adapt_prop:
            Sl = np.sum(Q_source_l)
            Sr = np.sum(Q_source_r)
            Slt = Y_target_node.size
            Srt = Y_target_node.size

            #Q_source_l = Q_source_l/np.sum(Q_source_l) 
            #Q_source_r = Q_source_r/np.sum(Q_source_r) 
            
            #Q_source_l = np.multiply(coeffs, Q_source_l)
            
            D = np.sum(np.multiply(coeffs, Q_source_l))
            Q_source_l = (Slt/Sl)*np.multiply(coeffs,np.divide(Q_source_l,D))
            D = np.sum(np.multiply(coeffs, Q_source_r))
            Q_source_r = (Srt/Sr)*np.multiply(coeffs,np.divide(Q_source_r,D))            
            
            #Q_source_r = np.multiply(coeffs, Q_source_r)
            #D = np.sum(Q_source_r)
            #Q_source_r = Sr*np.divide(Q_source_r,D)    
            
        Q_source_parent = lib_tree.get_node_distribution(decisiontree,
                                                node_index)

        # print("threshold selection : X_target_node shape : ",
        # X_target_node.shape)
        t1 = threshold_selection(Q_source_parent,
                                 Q_source_l.copy(),
                                 Q_source_r.copy(),
                                 X_target_node,
                                 Y_target_node,
                                 phi,
                                 classes,
                                 use_divergence=use_divergence,
                                 measure_default_IG=measure_default_IG)
        Q_target_l, Q_target_r = lib_tree.compute_Q_children_target(X_target_node,
                                                           Y_target_node,
                                                           phi,
                                                           t1,
                                                           classes)
        DG_t1 = lib_tree.DG(Q_source_l.copy(),
                   Q_source_r.copy(),
                   Q_target_l,
                   Q_target_r)
        t2 = threshold_selection(Q_source_parent,
                                 Q_source_r.copy(),
                                 Q_source_l.copy(),
                                 X_target_node,
                                 Y_target_node,
                                 phi,
                                 classes,
                                 use_divergence=use_divergence,
                                 measure_default_IG=measure_default_IG)


            
        Q_target_l, Q_target_r = lib_tree.compute_Q_children_target(X_target_node,
                                                           Y_target_node,
                                                           phi,
                                                           t2,
                                                           classes)
        DG_t2 = lib_tree.DG(Q_source_r.copy(),
                   Q_source_l.copy(),
                   Q_target_l,
                   Q_target_r)
        

        if DG_t1 >= DG_t2:
            decisiontree.tree_.threshold[node_index] = t1
        else:
            decisiontree.tree_.threshold[node_index] = t2
            # swap children
            old_child_r_id = decisiontree.tree_.children_right[node_index]
            decisiontree.tree_.children_right[node_index] = decisiontree.tree_.children_left[node_index]
            decisiontree.tree_.children_left[node_index] = old_child_r_id

    if decisiontree.tree_.children_left[node_index] != -1:
        # Computing target data passing through node NOT updated
        index_X_child_l = X_target_node_noupdate[:, phi] <= old_threshold
        X_target_node_noupdate_l = X_target_node_noupdate[index_X_child_l, :]
        Y_target_node_noupdate_l = Y_target_node_noupdate[index_X_child_l]
        # Computing target data passing through node updated
        threshold = decisiontree.tree_.threshold[node_index]
        index_X_child_l = X_target_node[:, phi] <= threshold
        X_target_child_l = X_target_node[index_X_child_l, :]
        Y_target_child_l = Y_target_node[index_X_child_l]
        
        node_index = STRUT(decisiontree,
              decisiontree.tree_.children_left[node_index],
              X_target_child_l,
              Y_target_child_l,
              X_target_node_noupdate_l,
              Y_target_node_noupdate_l,
              pruning_updated_node=pruning_updated_node,
              no_prune_on_cl=no_prune_on_cl,
              cl_no_prune=cl_no_prune,
              adapt_prop=adapt_prop,
              simple_weights=simple_weights,
              coeffs=coeffs,
              use_divergence = use_divergence,
              measure_default_IG=measure_default_IG)
        
        ## IMPORTANT ##
        node_index,b = lib_tree.find_parent(decisiontree, node_index)

    if decisiontree.tree_.children_right[node_index] != -1:
        # Computing target data passing through node NOT updated
        index_X_child_r = X_target_node_noupdate[:, phi] > old_threshold
        X_target_node_noupdate_r = X_target_node_noupdate[index_X_child_r, :]
        Y_target_node_noupdate_r = Y_target_node_noupdate[index_X_child_r]
        # Computing target data passing through node updated
        threshold = decisiontree.tree_.threshold[node_index]
        index_X_child_r = X_target_node[:, phi] > threshold
        X_target_child_r = X_target_node[index_X_child_r, :]
        Y_target_child_r = Y_target_node[index_X_child_r]

        node_index = STRUT(decisiontree,
              decisiontree.tree_.children_right[node_index],
              X_target_child_r,
              Y_target_child_r,
              X_target_node_noupdate_r,
              Y_target_node_noupdate_r,
              pruning_updated_node=pruning_updated_node,
              no_prune_on_cl=no_prune_on_cl,
              cl_no_prune=cl_no_prune,
              adapt_prop=adapt_prop,
              simple_weights=simple_weights,
              coeffs=coeffs,
              use_divergence=use_divergence,
              measure_default_IG=measure_default_IG)

        ## IMPORTANT ##
        node_index,b = lib_tree.find_parent(decisiontree, node_index)
        
    return node_index


def STRUT_RF(random_forest,
             X_target,
             y_target,
             pruning_updated_node=True,
             no_prune_on_cl=False,
             cl_no_prune=None,
             prune_lone_instance=True,
             adapt_prop=False,
             simple_weights=False,
             use_divergence=True,
             measure_default_IG=True):

    rf_strut = copy.deepcopy(random_forest)
    for i, dtree in enumerate(rf_strut.estimators_):

        if adapt_prop or simple_weights:
            props_s = lib_tree.get_node_distribution(rf_strut.estimators_[i], 0)
            props_s = props_s / sum(props_s)
            props_t = np.zeros(props_s.size)
            
            for k in range(props_s.size):
                props_t[k] = np.sum(y_target == k) / y_target.size
                
            coeffs = np.divide(props_t, props_s)
                


            #print("tree : ", i)
            STRUT(rf_strut.estimators_[i],
                  0,
                  X_target,
                  y_target,
                  X_target,
                  y_target,
                  pruning_updated_node=pruning_updated_node,
                  no_prune_on_cl=no_prune_on_cl,
                  cl_no_prune=cl_no_prune,
                  prune_lone_instance=prune_lone_instance,
                  adapt_prop=adapt_prop,
                  simple_weights=simple_weights,
                  coeffs=coeffs,
                  use_divergence=use_divergence,
                  measure_default_IG=measure_default_IG)
        else:
            STRUT(rf_strut.estimators_[i],
                  0,
                  X_target,
                  y_target,
                  X_target,
                  y_target,
                  pruning_updated_node=pruning_updated_node,
                  no_prune_on_cl=no_prune_on_cl,
                  cl_no_prune=cl_no_prune,
                  prune_lone_instance=prune_lone_instance,
                  use_divergence=use_divergence,
                  measure_default_IG=measure_default_IG)

    return rf_strut

    
if __name__ == "__main__":
    print('TEST :')
    import sys
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
        

    
    
    #import matplotlib.pyplot as plt
    #import seaborn as sns
    #from sklearn.tree import export_graphviz
    #X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_I()
    X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_6()
     
    MAX = 5
    solo_tree = False


    def true_pos(clf,X,y):
        return sum(clf.predict(X[y==1]) == 1)/sum(y==1)
    def false_pos(clf,X,y):
        return sum(clf.predict(X[y==0]) == 1)/sum(y==0)    
    
    if solo_tree:

        dtree_or = DecisionTreeClassifier(max_depth=MAX)
        dtree_or.fit(X_source,y_source)
    
        dts = np.zeros(3,dtype=object)

        cl_no_red = [1]
        Nkmin = sum(y_target_005 == cl_no_red )
        root_source_values = lib_tree.get_node_distribution(dtree_or, 0).reshape(-1)

        props_s = root_source_values
        props_s = props_s / sum(props_s)
        props_t = np.zeros(props_s.size)
        for k in range(props_s.size):
            props_t[k] = np.sum(y_target_005 == k) / y_target_005.size

        coeffs = np.divide(props_t, props_s)        
        dts[0] = copy.deepcopy(dtree_or)
        STRUT(dts[0],0, X_target_005, y_target_005, X_target_005, y_target_005)    
    
        dts[1] = copy.deepcopy(dtree_or)
        STRUT(dts[1],0, X_target_005, y_target_005, X_target_005, y_target_005, use_divergence=False)
        
        dts[2] = copy.deepcopy(dtree_or)
        STRUT(dts[2],0, X_target_005, y_target_005, X_target_005, y_target_005, adapt_prop=True, coeffs=coeffs)   
    

        
        netoile = 2
        print('score strut:', dts[0].score(X_target_095,y_target_095))
        print('score strut no div:', dts[1].score(X_target_095,y_target_095))
        print('score strut*:', dts[netoile].score(X_target_095,y_target_095))

        print('tpr strut:', true_pos(dts[0],X_target_095,y_target_095))
        print('tpr strut no div:', true_pos(dts[1],X_target_095,y_target_095))
        print('tpr strut*:', true_pos(dts[netoile],X_target_095,y_target_095))

        print('fpr strut:', false_pos(dts[0],X_target_095,y_target_095))
        print('fpr strut no div:', false_pos(dts[1],X_target_095,y_target_095))
        print('fpr strut*:', false_pos(dts[netoile],X_target_095,y_target_095))
        
        print('nb feuilles strut :',sum(dts[0].tree_.feature == -2))
        print('nb feuilles strut*:',sum(dts[netoile].tree_.feature == -2))
    else:
        
        N_EST = 3

        rf_or = RandomForestClassifier(n_estimators = N_EST,max_depth=MAX )
        rf_or.fit(X_source,y_source)
    
        rfs = np.zeros(13,dtype=object)

        rfs[0] = STRUT_RF(rf_or, X_target_005, y_target_005)              
        rfs[1] = STRUT_RF(rf_or, X_target_005, y_target_005, use_divergence=False)      
        rfs[2] = STRUT_RF(rf_or, X_target_005, y_target_005, adapt_prop=True)      

        netoile = 2
        dt = rfs[0].estimators_[0]
        dt_strut_adapt = rfs[netoile].estimators_[0]
        
        print('score strut:', rfs[0].score(X_target_095,y_target_095))
        print('score strut no div:', rfs[1].score(X_target_095,y_target_095))
        print('score strut*:', rfs[netoile].score(X_target_095,y_target_095))

        print('tpr strut:', true_pos(rfs[0],X_target_095,y_target_095))
        print('tpr strut no div:', true_pos(rfs[1],X_target_095,y_target_095))
        print('tpr strut*:', true_pos(rfs[netoile],X_target_095,y_target_095))

        print('fpr strut:', false_pos(rfs[0],X_target_095,y_target_095))
        print('fpr strut no div:', false_pos(rfs[1],X_target_095,y_target_095))
        print('fpr strut*:', false_pos(rfs[netoile],X_target_095,y_target_095))
        
        print('nb feuilles strut :',sum(dt.tree_.feature == -2))
        print('nb feuilles strut*:',sum(dt_strut_adapt.tree_.feature == -2))
