#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:52:46 2019

@author: mounir
"""


import os,sys
import numpy as np

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
          adapt_prop=False,
          coeffs=[1, 1],
          use_divergence=True,
          measure_default_IG=True):

    phi = decisiontree.tree_.feature[node_index]
    classes = decisiontree.classes_
    threshold = decisiontree.tree_.threshold[node_index]
    #old_threshold = threshold

    current_class_distribution = lib_tree.compute_class_distribution(classes, Y_target_node)

    decisiontree.tree_.weighted_n_node_samples[node_index] = Y_target_node.size
    decisiontree.tree_.impurity[node_index] = lib_tree.GINI(current_class_distribution)
    decisiontree.tree_.n_node_samples[node_index] = Y_target_node.size

    # If it is a leaf one, exit
    if decisiontree.tree_.children_left[node_index] == -2:

        decisiontree.tree_.value[node_index] = current_class_distribution
        return node_index

    is_reached_update = (current_class_distribution.sum() != 0)
    prune_cond = not is_reached_update


    if prune_cond:

        p,b = lib_tree.find_parent(decisiontree, node_index)
        node_index = lib_tree.cut_from_left_right(decisiontree,p,b)
        
        return node_index

    # else:
        # print("NO Pruning at node ", node_index)
        # return 0

    # update tree.value with target data
    decisiontree.tree_.value[node_index] = current_class_distribution

    # Only one class is present in the node -> terminal leaf
    if (current_class_distribution > 0).sum() == 1:

        node_index = lib_tree.cut_into_leaf2(decisiontree, node_index)
        return node_index
    

    # update threshold
    if type(threshold) is np.float64:
        Q_source_l, Q_source_r = lib_tree.get_children_distributions(decisiontree,
                                                                node_index)
        Sl = np.sum(Q_source_l)
        Sr = np.sum(Q_source_r)

   
        if adapt_prop:
            Sl = np.sum(Q_source_l)
            Sr = np.sum(Q_source_r)
            Slt = Y_target_node.size
            Srt = Y_target_node.size

            
            D = np.sum(np.multiply(coeffs, Q_source_l))
            Q_source_l = (Slt/Sl)*np.multiply(coeffs,np.divide(Q_source_l,D))
            D = np.sum(np.multiply(coeffs, Q_source_r))
            Q_source_r = (Srt/Sr)*np.multiply(coeffs,np.divide(Q_source_r,D))            
              
            
        Q_source_parent = lib_tree.get_node_distribution(decisiontree,
                                                node_index)


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
        #index_X_child_l = X_target_node_noupdate[:, phi] <= old_threshold
        #X_target_node_noupdate_l = X_target_node_noupdate[index_X_child_l, :]
        #Y_target_node_noupdate_l = Y_target_node_noupdate[index_X_child_l]
        # Computing target data passing through node updated
        threshold = decisiontree.tree_.threshold[node_index]
        index_X_child_l = X_target_node[:, phi] <= threshold
        X_target_child_l = X_target_node[index_X_child_l, :]
        Y_target_child_l = Y_target_node[index_X_child_l]
        
        node_index = STRUT(decisiontree,
              decisiontree.tree_.children_left[node_index],
              X_target_child_l,
              Y_target_child_l,
              adapt_prop=adapt_prop,
              coeffs=coeffs,
              use_divergence = use_divergence,
              measure_default_IG=measure_default_IG)
        
        ## IMPORTANT ##
        node_index,b = lib_tree.find_parent(decisiontree, node_index)

    if decisiontree.tree_.children_right[node_index] != -1:
        # Computing target data passing through node NOT updated
        #index_X_child_r = X_target_node_noupdate[:, phi] > old_threshold
        #X_target_node_noupdate_r = X_target_node_noupdate[index_X_child_r, :]
        #Y_target_node_noupdate_r = Y_target_node_noupdate[index_X_child_r]
        # Computing target data passing through node updated
        threshold = decisiontree.tree_.threshold[node_index]
        index_X_child_r = X_target_node[:, phi] > threshold
        X_target_child_r = X_target_node[index_X_child_r, :]
        Y_target_child_r = Y_target_node[index_X_child_r]

        node_index = STRUT(decisiontree,
              decisiontree.tree_.children_right[node_index],
              X_target_child_r,
              Y_target_child_r,
              adapt_prop=adapt_prop,
              coeffs=coeffs,
              use_divergence=use_divergence,
              measure_default_IG=measure_default_IG)

        ## IMPORTANT ##
        node_index,b = lib_tree.find_parent(decisiontree, node_index)
        
    return node_index


def STRUT_RF(random_forest,
             X_target,
             y_target,
             adapt_prop=False,
             use_divergence=True,
             measure_default_IG=True):

    rf_strut = copy.deepcopy(random_forest)
    for i, dtree in enumerate(rf_strut.estimators_):

        if adapt_prop:
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
                  adapt_prop=adapt_prop,
                  coeffs=coeffs,
                  use_divergence=use_divergence,
                  measure_default_IG=measure_default_IG)
        else:
            STRUT(rf_strut.estimators_[i],
                  0,
                  X_target,
                  y_target,
                  use_divergence=use_divergence,
                  measure_default_IG=measure_default_IG)

    return rf_strut

    
#if __name__ == "__main__":
#    print('TEST :')
