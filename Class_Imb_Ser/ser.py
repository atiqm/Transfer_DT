#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:44:35 2019

@author: mounir
"""

import os,sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import copy


sys.path.insert(0,'../')
import lib_tree


def SER(node, dTree, X_target_node, y_target_node, original_ser=True, no_red_on_cl=False,
        cl_no_red=None, no_ext_on_cl=False, cl_no_ext=None, ext_cond=None, leaf_loss_quantify = False, leaf_loss_threshold = None, coeffs = None, root_source_values = None, Nkmin = None ):
    

    # Deep copy of value
    old_values = dTree.tree_.value[node].copy()
    maj_class = np.argmax(dTree.tree_.value[node, :].copy())
           
    if cl_no_red is None:
        old_size_cl_no_red = 0
    else:
        old_size_cl_no_red = np.sum(dTree.tree_.value[node][:, cl_no_red])

    if no_red_on_cl is not None or no_ext_on_cl is not None :
        if no_ext_on_cl:
            cl = cl_no_ext[0] 
        if no_red_on_cl:
            cl = cl_no_red[0]
        
    if leaf_loss_quantify and ((no_red_on_cl  or  no_ext_on_cl) and maj_class == cl) and  dTree.tree_.feature[node] == -2 :
            
        ps_rf = dTree.tree_.value[node,0,:]/sum(dTree.tree_.value[node,0,:])                
        p1_in_l = dTree.tree_.value[node,0,cl]/root_source_values[cl]
        cond1 = np.power(1 - p1_in_l,Nkmin) > leaf_loss_threshold
        cond2 = np.argmax(np.multiply(coeffs,ps_rf)) == cl
         

    ### VALUES UPDATE ###
    val = np.zeros((dTree.n_outputs_, dTree.n_classes_))

    for i in range(dTree.n_classes_):
        val[:, i] = list(y_target_node).count(i)
        
    dTree.tree_.value[node] = val
    dTree.tree_.n_node_samples[node] = np.sum(val)
    dTree.tree_.weighted_n_node_samples[node] = np.sum(val)        
        
    if dTree.tree_.feature[node]== -2:
        if original_ser:
            if y_target_node.size > 0 and len(set(list(y_target_node))) > 1:
                #la classe change automatiquement en fonction de target par les values updates

                DT_to_add = DecisionTreeClassifier()

                try:
                    DT_to_add.min_impurity_decrease = 0
                except:
                    DT_to_add.min_impurity_split = 0
                DT_to_add.fit(X_target_node, y_target_node)
                lib_tree.fusionDecisionTree(dTree, node, DT_to_add)
                    
            return node,False
        
        else:
            bool_no_red = False
            cond_extension = False
            
            if y_target_node.size > 0:
                #Extension
                if not no_ext_on_cl:
                    DT_to_add = DecisionTreeClassifier()
                    # to make a complete tree
                    try:
                        DT_to_add.min_impurity_decrease = 0
                    except:
                        DT_to_add.min_impurity_split = 0
                    DT_to_add.fit(X_target_node, y_target_node)
                    lib_tree.fusionDecisionTree(dTree, node, DT_to_add)
                else:
                    cond_maj = (maj_class not in cl_no_ext)
                    cond_sub_target = ext_cond and (maj_class in y_target_node) and (maj_class in cl_no_ext)
                    cond_leaf_loss = leaf_loss_quantify and maj_class==cl and not (cond1 and cond2)
                    
                    cond_extension = cond_maj or cond_sub_target or cond_leaf_loss
                    
                    if cond_extension:
                        DT_to_add = DecisionTreeClassifier()
                        # to make a complete tree
                        try:
                            DT_to_add.min_impurity_decrease = 0
                        except:
                            DT_to_add.min_impurity_split = 0
                        DT_to_add.fit(X_target_node, y_target_node)
                        lib_tree.fusionDecisionTree(dTree, node, DT_to_add)
                    else:
                        ## Compliqué de ne pas induire d'incohérence au niveau des values
                        ## en laissant intactes les feuilles de cette manière.
                        ## Cela dit, ça n'a pas d'impact sur l'arbre décisionnel qu'on veut
                        ## obtenir (ça en a un sur l'arbre probabilisé)
                        dTree.tree_.value[node] = old_values
                        dTree.tree_.n_node_samples[node] = np.sum(old_values)
                        dTree.tree_.weighted_n_node_samples[
                            node] = np.sum(old_values)
                        lib_tree.add_to_parents(dTree, node, old_values)
                        if no_red_on_cl:
                            bool_no_red = True
            
            #no red protection with values
            if no_red_on_cl and y_target_node.size == 0 and old_size_cl_no_red > 0 and maj_class in cl_no_red:
                
                if leaf_loss_quantify :

                    if cond1 and cond2 :
                        dTree.tree_.value[node] = old_values
                        dTree.tree_.n_node_samples[node] = np.sum(old_values)
                        dTree.tree_.weighted_n_node_samples[node] = np.sum(old_values)
                        lib_tree.add_to_parents(dTree, node, old_values)
                        bool_no_red = True
                else:
                    dTree.tree_.value[node] = old_values
                    dTree.tree_.n_node_samples[node] = np.sum(old_values)
                    dTree.tree_.weighted_n_node_samples[node] = np.sum(old_values)
                    lib_tree.add_to_parents(dTree, node, old_values)
                    bool_no_red = True 
                    
            return node,bool_no_red

    ### Left / right target computation ###
    bool_test = X_target_node[:, dTree.tree_.feature[node]] <= dTree.tree_.threshold[node]
    not_bool_test = X_target_node[:, dTree.tree_.feature[node]] > dTree.tree_.threshold[node]

    ind_left = np.where(bool_test)[0]
    ind_right = np.where(not_bool_test)[0]

    X_target_node_left = X_target_node[ind_left]
    y_target_node_left = y_target_node[ind_left]

    X_target_node_right = X_target_node[ind_right]
    y_target_node_right = y_target_node[ind_right]     
    
    if original_ser:
        new_node_left,bool_no_red_l = SER(dTree.tree_.children_left[node], dTree, X_target_node_left, y_target_node_left,
                                      original_ser = True)        
        node, b = lib_tree.find_parent(dTree, new_node_left)

        new_node_right,bool_no_red_r = SER(dTree.tree_.children_right[node], dTree, X_target_node_right, y_target_node_right,
                                      original_ser = True)         
        node, b = lib_tree.find_parent(dTree, new_node_right)

    else:
        new_node_left,bool_no_red_l = SER(dTree.tree_.children_left[node], dTree, X_target_node_left, y_target_node_left,original_ser=False,
                            no_red_on_cl=no_red_on_cl, cl_no_red=cl_no_red,
                            no_ext_on_cl=no_ext_on_cl, cl_no_ext=cl_no_ext, leaf_loss_quantify=leaf_loss_quantify,
                            leaf_loss_threshold=leaf_loss_threshold, coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)
    
        node, b = lib_tree.find_parent(dTree, new_node_left)
        
        new_node_right,bool_no_red_r = SER(dTree.tree_.children_right[node], dTree, X_target_node_right, y_target_node_right,original_ser=False,
                             no_red_on_cl=no_red_on_cl, cl_no_red=cl_no_red,
                             no_ext_on_cl=no_ext_on_cl, cl_no_ext=cl_no_ext, leaf_loss_quantify=leaf_loss_quantify, 
                             leaf_loss_threshold=leaf_loss_threshold, coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)

        node, b = lib_tree.find_parent(dTree, new_node_right)
        
    if original_ser:
        bool_no_red = False
    else:
        bool_no_red = bool_no_red_l or bool_no_red_r
    
    
    le = lib_tree.leaf_error(dTree.tree_, node)
    e = lib_tree.error(dTree.tree_, node)

    if le <= e:
        if original_ser:
            new_node_leaf = lib_tree.cut_into_leaf2(dTree, node)
            node = new_node_leaf
        else:
            if no_red_on_cl:
                if not bool_no_red:
                    new_node_leaf = lib_tree.cut_into_leaf2(dTree, node)
                    node = new_node_leaf
#                else: 
#                    print('avoid pruning')
            else:
                new_node_leaf = lib_tree.cut_into_leaf2(dTree, node)
                node = new_node_leaf
   
    if dTree.tree_.feature[node] != -2:
        if original_ser:
            if ind_left.size == 0:
                node = lib_tree.cut_from_left_right(dTree, node, -1)

            if ind_right.size == 0:
                node = lib_tree.cut_from_left_right(dTree, node, 1)
        else:
            if no_red_on_cl:
                if ind_left.size == 0 and np.sum(dTree.tree_.value[dTree.tree_.children_left[node]]) == 0:
                    node = lib_tree.cut_from_left_right(dTree, node, -1)
    
                if ind_right.size == 0 and np.sum(dTree.tree_.value[dTree.tree_.children_right[node]]) == 0:
                    node = lib_tree.cut_from_left_right(dTree, node, 1)
            else:
                if ind_left.size == 0:
                    node = lib_tree.cut_from_left_right(dTree, node, -1)
    
                if ind_right.size == 0:
                    node = lib_tree.cut_from_left_right(dTree, node, 1)



    
    return node, bool_no_red
    
def SER_RF(random_forest, X_target, y_target, original_ser=True, bootstrap_=False,
           no_red_on_cl=False, cl_no_red=None, no_ext_on_cl=False, cl_no_ext=None,
           ext_cond=False, leaf_loss_quantify = False, leaf_loss_threshold = 0.9):
    
    rf_ser = copy.deepcopy(random_forest)
    
    for i, dtree in enumerate(rf_ser.estimators_):
        root_source_values = None
        coeffs = None
        Nkmin = None
        if  leaf_loss_quantify :    
            Nkmin = sum(y_target == cl_no_red )
            root_source_values = lib_tree.get_node_distribution(rf_ser.estimators_[i], 0).reshape(-1)

            props_s = root_source_values
            props_s = props_s / sum(props_s)
            props_t = np.zeros(props_s.size)
            for k in range(props_s.size):
                props_t[k] = np.sum(y_target == k) / y_target.size
            
            coeffs = np.divide(props_t, props_s)
                            
        inds = np.linspace(0, y_target.size - 1, y_target.size).astype(int)
        if bootstrap_:
            inds = bootstrap(y_target.size)

        SER(0, rf_ser.estimators_[i], X_target[inds], y_target[inds],original_ser=original_ser,
            no_red_on_cl=no_red_on_cl, cl_no_red=cl_no_red,
            no_ext_on_cl=no_ext_on_cl, cl_no_ext=cl_no_ext,
            ext_cond=ext_cond,leaf_loss_quantify=leaf_loss_quantify,leaf_loss_threshold=leaf_loss_threshold,coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)
        
    return rf_ser
    

def bootstrap(size):
    return np.random.choice(np.linspace(0, size - 1, size).astype(int), size, replace=True)

# =============================================================================
# 
# =============================================================================


#if __name__ == "__main__":
#    print('TEST :')
