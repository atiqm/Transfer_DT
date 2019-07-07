#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:44:35 2019

@author: mounir
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from STRUT import get_node_distribution
import copy



# =============================================================================
# 
# =============================================================================
### A bien organiser plus tard

def depth_vtree(tree,node):
    p,t,b = extract_rule_vtree(tree,node)
    return len(p)

def find_parent_vtree(tree, i_node):
    p = -1
    b = 0
    if i_node != 0 and i_node != -1:

        try:
            p = list(tree.children_left).index(i_node)
            b = -1
        except:
            p = p
        try:
            p = list(tree.children_right).index(i_node)
            b = 1
        except:
            p = p

    return p, b

def extract_rule_vtree(tree,node):

    feats = list()
    ths = list()
    bools = list()
    nodes = list()
    b = 1
    if node != 0:
        while b != 0:
           
            feats.append(tree.feature[node])
            ths.append(tree.threshold[node])
            bools.append(b)
            nodes.append(node)
            node,b = find_parent_vtree(tree,node)
            
        feats.pop(0)
        ths.pop(0)
        bools.pop(0)
        nodes.pop(0)
  
    return np.array(feats), np.array(ths), np.array(bools)
# =============================================================================
# 
# =============================================================================

def depth_rf(rf):
    d = 0
    for p in rf.estimators_:
        d = d + p.tree_.max_depth
    return d/len(rf.estimators_)

def depth(dtree,node):
    p,t,b = extract_rule(dtree,node)
    return len(p)

def depth_array(dtree, inds):
    depths = np.zeros(np.array(inds).size)
    for i, e in enumerate(inds):
        depths[i] = depth(dtree, i)
    return depths


def leaf_error(tree, node):
    if np.sum(tree.value[node]) == 0:
        return 0
    else:
        return 1 - np.max(tree.value[node]) / np.sum(tree.value[node])


def error(tree, node):
    if node == -1:
        return 0
    else:

        if tree.feature[node] == -2:
            return leaf_error(tree, node)
        else:
            # Pas une feuille

            nr = np.sum(tree.value[tree.children_right[node]])
            nl = np.sum(tree.value[tree.children_left[node]])

            if nr + nl == 0:
                return 0
            else:
                er = error(tree, tree.children_right[node])
                el = error(tree, tree.children_left[node])

                return (el * nl + er * nr) / (nl + nr)
# =============================================================================
# 
# =============================================================================
def extract_rule(dtree,node):

    feats = list()
    ths = list()
    bools = list()
    nodes = list()
    b = 1
    if node != 0:
        while b != 0:
           
            feats.append(dtree.tree_.feature[node])
            ths.append(dtree.tree_.threshold[node])
            bools.append(b)
            nodes.append(node)
            node,b = find_parent(dtree,node)
            
        feats.pop(0)
        ths.pop(0)
        bools.pop(0)
        nodes.pop(0)
  
    return np.array(feats), np.array(ths), np.array(bools)


def extract_leaves_rules(dtree):
    leaves = np.where(dtree.tree_.feature == -2)[0]
    
    rules = np.zeros(leaves.size,dtype = object)
    for k,f in enumerate(leaves) :
        rules[k] = extract_rule(dtree,f)
        
    return leaves, rules

def find_parent(dtree, i_node):
    p = -1
    b = 0
    if i_node != 0 and i_node != -1:

        try:
            p = list(dtree.tree_.children_left).index(i_node)
            b = -1
        except:
            p = p
        try:
            p = list(dtree.tree_.children_right).index(i_node)
            b = 1
        except:
            p = p

    return p, b


#def find_parent_dic(dic, i_node):
#    p = -1
#    b = 0
#    if i_node != 0 and i_node != -1:
#
#        try:
#            p = list(dic['nodes']['left_child']).index(i_node)
#            b = -1
#        except:
#            p = p
#        try:
#            p = list(dic['nodes']['right_child']).index(i_node)
#            b = 1
#        except:
#            p = p
#
#    return p, b


def sub_nodes(tree, node):
    if (node == -1):
        return list()
    if (tree.feature[node] == -2):
        return [node]
    else:
        return [node] + sub_nodes(tree, tree.children_left[node]) + sub_nodes(tree, tree.children_right[node])


# =============================================================================
#            
# =============================================================================

def fusionTree(tree1, f, tree2):
    """adding tree tree2 to leaf f of tree tree1"""

    dic = tree1.__getstate__().copy()
    dic2 = tree2.__getstate__().copy()

    size_init = tree1.node_count

    if depth_vtree(tree1, f) + dic2['max_depth'] > dic['max_depth']:
        dic['max_depth'] = depth_vtree(tree1, f) + tree2.max_depth

    dic['capacity'] = tree1.capacity + tree2.capacity - 1
    dic['node_count'] = tree1.node_count + tree2.node_count - 1

    dic['nodes'][f] = dic2['nodes'][0]

    if (dic2['nodes']['left_child'][0] != - 1):
        dic['nodes']['left_child'][f] = dic2[
            'nodes']['left_child'][0] + size_init - 1
    else:
        dic['nodes']['left_child'][f] = -1
    if (dic2['nodes']['right_child'][0] != - 1):
        dic['nodes']['right_child'][f] = dic2[
            'nodes']['right_child'][0] + size_init - 1
    else:
        dic['nodes']['right_child'][f] = -1

    # Attention vecteur impurity pas mis à jour

    dic['nodes'] = np.concatenate((dic['nodes'], dic2['nodes'][1:]))
    dic['nodes']['left_child'][size_init:] = (dic['nodes']['left_child'][
                                              size_init:] != -1) * (dic['nodes']['left_child'][size_init:] + size_init) - 1
    dic['nodes']['right_child'][size_init:] = (dic['nodes']['right_child'][
                                               size_init:] != -1) * (dic['nodes']['right_child'][size_init:] + size_init) - 1

    values = np.concatenate((dic['values'], np.zeros((dic2['values'].shape[
                            0] - 1, dic['values'].shape[1], dic['values'].shape[2]))), axis=0)

    dic['values'] = values

    # Attention :: (potentiellement important)
    (Tree, (n_f, n_c, n_o), b) = tree1.__reduce__()
    #del tree1
    #del tree2

    tree1 = Tree(n_f, n_c, n_o)

    tree1.__setstate__(dic)
    return tree1


def fusionDecisionTree(dTree1, f, dTree2):
    """adding tree dTree2 to leaf f of tree dTree1"""
    #dTree = sklearn.tree.DecisionTreeClassifier()
    size_init = dTree1.tree_.node_count
    dTree1.tree_ = fusionTree(dTree1.tree_, f, dTree2.tree_)

    try:
        dTree1.tree_.value[size_init:, :, dTree2.classes_.astype(
            int)] = dTree2.tree_.value[1:, :, :]
    except IndexError as e:
        print("IndexError : size init : ", size_init,
              "\ndTree2.classes_ : ", dTree2.classes_)
        print(e)
    dTree1.max_depth = dTree1.tree_.max_depth
    return dTree1

def cut_from_left_right(dTree, node, bool_left_right):
    dic = dTree.tree_.__getstate__().copy()

    node_to_rem = list()
    size_init = dTree.tree_.node_count

    p, b = find_parent(dTree, node)

    if bool_left_right == 1:
        repl_node = dTree.tree_.children_left[node]
        #node_to_rem = [node] + sub_nodes(dTree.tree_,dTree.tree_.children_right[node])
        node_to_rem = [node, dTree.tree_.children_right[node]]
    elif bool_left_right == -1:
        repl_node = dTree.tree_.children_right[node]
        #node_to_rem = [node] + sub_nodes(dTree.tree_,dTree.tree_.children_left[node])
        node_to_rem = [node, dTree.tree_.children_left[node]]

    inds = list(
        set(np.linspace(0, size_init - 1, size_init).astype(int)) - set(node_to_rem))

    dic['capacity'] = dTree.tree_.capacity - len(node_to_rem)
    dic['node_count'] = dTree.tree_.node_count - len(node_to_rem)

    if b == 1:
        dic['nodes']['right_child'][p] = repl_node
    elif b == -1:
        dic['nodes']['left_child'][p] = repl_node

    #new_size = len(ind)
    dic_old = dic.copy()
    left_old = dic_old['nodes']['left_child']
    right_old = dic_old['nodes']['right_child']
    #print('taille avant:',dic['nodes'].size)
    dic['nodes'] = dic['nodes'][inds]
    dic['values'] = dic['values'][inds]
    #print('taille après:',dic['nodes'].size)

    for i, new in enumerate(inds):
        if (left_old[new] != -1):
            dic['nodes']['left_child'][i] = inds.index(left_old[new])
        else:
            dic['nodes']['left_child'][i] = -1
        if (right_old[new] != -1):
            dic['nodes']['right_child'][i] = inds.index(right_old[new])
        else:
            dic['nodes']['right_child'][i] = -1

    (Tree, (n_f, n_c, n_o), b) = dTree.tree_.__reduce__()
    del dTree.tree_

    dTree.tree_ = Tree(n_f, n_c, n_o)
    dTree.tree_.__setstate__(dic)
    depths = depth_array(dTree, np.linspace(
        0, dTree.tree_.node_count - 1, dTree.tree_.node_count).astype(int))
    dTree.tree_.max_depth = np.max(depths)

    return inds.index(repl_node)


def cut_into_leaf2(dTree, node):
    dic = dTree.tree_.__getstate__().copy()

    node_to_rem = list()
    size_init = dTree.tree_.node_count

    node_to_rem = node_to_rem + sub_nodes(dTree.tree_, node)[1:]
    node_to_rem = list(set(node_to_rem))

    inds = list(
        set(np.linspace(0, size_init - 1, size_init).astype(int)) - set(node_to_rem))
    depths = depth_array(dTree, inds)
    dic['max_depth'] = np.max(depths)

    dic['capacity'] = dTree.tree_.capacity - len(node_to_rem)
    dic['node_count'] = dTree.tree_.node_count - len(node_to_rem)

    dic['nodes']['feature'][node] = -2
    dic['nodes']['left_child'][node] = -1
    dic['nodes']['right_child'][node] = -1

    #new_size = len(ind)
    dic_old = dic.copy()
    left_old = dic_old['nodes']['left_child']
    right_old = dic_old['nodes']['right_child']
    #print('taille avant:',dic['nodes'].size)
    dic['nodes'] = dic['nodes'][inds]
    dic['values'] = dic['values'][inds]
    #print('taille après:',dic['nodes'].size)

    for i, new in enumerate(inds):
        if (left_old[new] != -1):
            dic['nodes']['left_child'][i] = inds.index(left_old[new])
        else:
            dic['nodes']['left_child'][i] = -1
        if (right_old[new] != -1):
            dic['nodes']['right_child'][i] = inds.index(right_old[new])
        else:
            dic['nodes']['right_child'][i] = -1

    (Tree, (n_f, n_c, n_o), b) = dTree.tree_.__reduce__()
    del dTree.tree_

    dTree.tree_ = Tree(n_f, n_c, n_o)
    dTree.tree_.__setstate__(dic)

    return inds.index(node)

def add_to_parents(dTree, node, values):

    p, b = find_parent(dTree, node)

    if b != 0:
        dTree.tree_.value[p] = dTree.tree_.value[p] + values
        add_to_parents(dTree, p, values)


def add_to_child(dTree, node, values):

    l = dTree.tree_.children_left[node]
    r = dTree.tree_.children_right[node]

    if r != -1:
        dTree.tree_.value[r] = dTree.tree_.value[r] + values
        add_to_child(dTree, r, values)
    if l != -1:
        dTree.tree_.value[l] = dTree.tree_.value[l] + values
        add_to_child(dTree, l, values)
        
# =============================================================================
# 
# =============================================================================



def SER(node, dTree, X_target_node, y_target_node, original_ser=True, no_red_on_cl=False,
        cl_no_red=None, no_ext_on_cl=False, cl_no_ext=None, ext_cond=None, leaf_loss_quantify = False, leaf_loss_threshold = None, coeffs = None, root_source_values = None, Nkmin = None ):
    

    # CARE : Deep copy of value
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
        
        if cond1 and not cond2:
            print('coucou')
            print('cond1 : ',cond1)
            print('cond2 : ',cond2)
            print( dTree.tree_.value[node,0,:])  

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
                fusionDecisionTree(dTree, node, DT_to_add)
                    
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
                    fusionDecisionTree(dTree, node, DT_to_add)
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
                        fusionDecisionTree(dTree, node, DT_to_add)
                    else:
                        ## Compliqué de ne pas induire d'incohérence au niveau des values
                        ## en laissant intactes les feuilles de cette manière.
                        ## Cela dit, ça n'a pas d'impact sur l'arbre décisionnel qu'on veut
                        ## obtenir (ça en a un sur l'arbre probabilisé)
                        dTree.tree_.value[node] = old_values
                        dTree.tree_.n_node_samples[node] = np.sum(old_values)
                        dTree.tree_.weighted_n_node_samples[
                            node] = np.sum(old_values)
                        add_to_parents(dTree, node, old_values)
                        if no_red_on_cl:
                            bool_no_red = True
            
            #no red protection with values

            if no_red_on_cl and y_target_node.size == 0 and old_size_cl_no_red > 0 and maj_class in cl_no_red:
                
                if leaf_loss_quantify :

                    if cond1 and cond2 :
                        dTree.tree_.value[node] = old_values
                        dTree.tree_.n_node_samples[node] = np.sum(old_values)
                        dTree.tree_.weighted_n_node_samples[node] = np.sum(old_values)
                        add_to_parents(dTree, node, old_values)
                        bool_no_red = True
                else:
                    dTree.tree_.value[node] = old_values
                    dTree.tree_.n_node_samples[node] = np.sum(old_values)
                    dTree.tree_.weighted_n_node_samples[node] = np.sum(old_values)
                    add_to_parents(dTree, node, old_values)
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
        node, b = find_parent(dTree, new_node_left)

        new_node_right,bool_no_red_r = SER(dTree.tree_.children_right[node], dTree, X_target_node_right, y_target_node_right,
                                      original_ser = True)         
        node, b = find_parent(dTree, new_node_right)

    else:
        new_node_left,bool_no_red_l = SER(dTree.tree_.children_left[node], dTree, X_target_node_left, y_target_node_left,original_ser=False,
                            no_red_on_cl=no_red_on_cl, cl_no_red=cl_no_red,
                            no_ext_on_cl=no_ext_on_cl, cl_no_ext=cl_no_ext, leaf_loss_quantify=leaf_loss_quantify,
                            leaf_loss_threshold=leaf_loss_threshold, coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)
        
        #dic = dTree.tree_.__getstate__().copy()
        #node, b = find_parent_dic(dic, new_node_left)
    
        node, b = find_parent(dTree, new_node_left)
        
        new_node_right,bool_no_red_r = SER(dTree.tree_.children_right[node], dTree, X_target_node_right, y_target_node_right,original_ser=False,
                             no_red_on_cl=no_red_on_cl, cl_no_red=cl_no_red,
                             no_ext_on_cl=no_ext_on_cl, cl_no_ext=cl_no_ext, leaf_loss_quantify=leaf_loss_quantify, 
                             leaf_loss_threshold=leaf_loss_threshold, coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)
      
    #dic = dTree.tree_.__getstate__().copy()
    #node, b = find_parent_dic(dic, new_node_right)
        node, b = find_parent(dTree, new_node_right)
        
    if original_ser:
        bool_no_red = False
    else:
        bool_no_red = bool_no_red_l or bool_no_red_r
    
    
    le = leaf_error(dTree.tree_, node)
    e = error(dTree.tree_, node)

    if le <= e:
        if original_ser:
            new_node_leaf = cut_into_leaf2(dTree, node)
            node = new_node_leaf
        else:
            if no_red_on_cl:
                if not bool_no_red:
                    new_node_leaf = cut_into_leaf2(dTree, node)
                    node = new_node_leaf
#                else: 
#                    print('avoid pruning')
            else:
                new_node_leaf = cut_into_leaf2(dTree, node)
                node = new_node_leaf
#    if le <= e:
#        new_node_leaf = cut_into_leaf2(dTree, node)
#        node = new_node_leaf         
#        
    if dTree.tree_.feature[node] != -2:
        if original_ser:
            if ind_left.size == 0:
                node = cut_from_left_right(dTree, node, -1)

            if ind_right.size == 0:
                node = cut_from_left_right(dTree, node, 1)
        else:
            if no_red_on_cl:
                if ind_left.size == 0 and np.sum(dTree.tree_.value[dTree.tree_.children_left[node]]) == 0:
                    node = cut_from_left_right(dTree, node, -1)
    
                if ind_right.size == 0 and np.sum(dTree.tree_.value[dTree.tree_.children_right[node]]) == 0:
                    node = cut_from_left_right(dTree, node, 1)
            else:
                if ind_left.size == 0:
                    node = cut_from_left_right(dTree, node, -1)
    
                if ind_right.size == 0:
                    node = cut_from_left_right(dTree, node, 1)



    
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
            root_source_values = get_node_distribution(rf_ser.estimators_[i], 0).reshape(-1)

            props_s = root_source_values
            props_s = props_s / sum(props_s)
            props_t = np.zeros(props_s.size)
            for k in range(props_s.size):
                props_t[k] = np.sum(y_target == k) / y_target.size
            
            coeffs = np.divide(props_t, props_s)
                
            #source_values_tot = rf_ser.estimators_[i].tree_.value[0,0,cl_no_red]
            
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


if __name__ == "__main__":
    print('TEST :')
    import sys
    sys.path.insert(0, "../data_mngmt/")
    sys.path.insert(0, "../utils/")
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.tree import export_graphviz
    from data import load_letter, load_wine, load_mushroom, load_mnist, load_6, load_T, load_I
#    X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_mnist()
#    X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_letter()
#    X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_I()
#    X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_T(1)


# =============================================================================
#             SYNTH
# =============================================================================
    sys.path.insert(0, "../synth_gen/")
    from Generator import ClusterPoints,StreamGenerator
    from Changes import apply_drift, delete_clusters, change_cluster_weight, create_new_clusters, apply_density_change
    import lib_synth_exp as lib
    def change_prop(sg,weights):
        labels = list(set(sg.class_labels))
        N = len(labels)
        
        nb_cl = np.zeros(N)
        
        for i in range(N):
            nb_cl[i] = sg.class_labels.count(labels[i])
            sg.weights[sg.class_labels == labels[i]] = weights[i]/nb_cl[i]
    
            clusts = list(np.array(sg.clusters)[sg.class_labels == labels[i]])
            for c in clusts:
                c.weight = weights[i]/nb_cl[i]


    bool_cluster_change = True
    del_cluster = [0,0]
    add_cluster = [10,10]
    
    drift = False
    var_change = True
    v_drift = 30
    m_drift = 15
    factor_var_m = 1
    factor_var_M = 4
        
    p0 = 0.95
    p1 = 1 - p0
# =============================================================================
#                 
# =============================================================================
    N1 = 15
    N0 = 15
    
    dim = 2
    space_bound = 50
    
    n_source = 500
    n_target_train = 100
    n_target_test = 1000
       
# =============================================================================
# 
# =============================================================================
    prop1 = 0.5
    prop1 = prop1/N1
    prop0 = 0.5
    prop0 = prop0/N0

    weights = list(np.concatenate((np.repeat(prop1,N1),np.repeat(prop0,N0))))
    class_labels = list(np.concatenate((np.repeat(1,N1),np.repeat(0,N0))))


    name_param = "std"
    sg = StreamGenerator(number_points=n_source,
    					 weights=weights,
    					 dimensionality=dim,
    					 min_projected_dim=dim,
    					 max_projected_dim=dim,
    					 min_coordinate=-space_bound,
    					 max_coordinate=space_bound,
    					 min_projected_dim_var=5,
    					 max_projected_dim_var=15,
    					 class_labels= class_labels)
    
    a = sg.get_full_dataset(n_source)
    X_source = a[1].values[:,:-1]
    y_source = a[1].values[:,-1]
    
# =============================================================================
# 
# =============================================================================

    
    if bool_cluster_change :
        if ( sum(np.array(del_cluster))> 0) :
            for i in range(len(del_cluster)):
                #ATTENTION:
                lib.del_cl_prop(sg,del_cluster[i],i,p0)
        if (sum(np.array(add_cluster))> 0 ) :
            for i in range(len(del_cluster)):
                lib.new_cl_prop(sg,add_cluster[i],i,p0)
    
    # Drift and Stretching : 
    if drift:
        lib.random_drift_cl(sg,v_drift,len(sg.class_labels),bias = m_drift)
    if var_change :
        lib.random_var_cl(sg,factor_var_m,factor_var_M,len(sg.class_labels))

    change_prop(sg,[p0,p1])
# =============================================================================
# 
# =============================================================================
    b = sg.get_full_dataset(n_target_train)
    X_target_005 = b[1].values[:,:-1]
    y_target_005 = b[1].values[:,-1]
        
    c = sg.get_full_dataset(n_target_test)
    X_target_095 = c[1].values[:,:-1]
    y_target_095 = c[1].values[:,-1]

# =============================================================================
# 
# =============================================================================
    plots = True 
    if plots:
        sns.pairplot( b[1],hue="cluster")
        sns.pairplot( c[1],hue="cluster")
        #plt.savefig(path_out+'train.pdf')
        plt.show()
        
    MAX = 5
    solo_tree = True
    if solo_tree:

        
        dtree_or = DecisionTreeClassifier(max_depth=MAX)

        dtree_or.fit(X_source,y_source)
    
        dts = np.zeros(13,dtype=object)

        cl_no_red = [1]
        Nkmin = sum(y_target_005 == cl_no_red )
        root_source_values = get_node_distribution(dtree_or, 0).reshape(-1)

        props_s = root_source_values
        props_s = props_s / sum(props_s)
        props_t = np.zeros(props_s.size)
        for k in range(props_s.size):
            props_t[k] = np.sum(y_target_005 == k) / y_target_005.size
        
        coeffs = np.divide(props_t, props_s)        
        dts[1] = copy.deepcopy(dtree_or)
        SER(0,dts[1], X_target_005, y_target_005, original_ser=True )    
    
        dts[2] = copy.deepcopy(dtree_or)
        SER(0,dts[2], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True,cl_no_red= cl_no_red)
        
        dts[3] = copy.deepcopy(dtree_or)
        SER(0,dts[3], X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext= cl_no_red )    
    
        #no ext vars
        dts[4] = copy.deepcopy(dtree_or)
        SER(0,dts[4], X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext= cl_no_red ,ext_cond=True)
        dts[5] = copy.deepcopy(dtree_or)
        SER(0,dts[5], X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext= cl_no_red ,leaf_loss_quantify=True, leaf_loss_threshold = 0.2,coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)    
        dts[6] = copy.deepcopy(dtree_or)
        SER(0,dts[6], X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext= cl_no_red ,ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.9,coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)    
        
        #no red vars
        dts[7] = copy.deepcopy(dtree_or)
        SER(0,dts[7], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=cl_no_red ,ext_cond=True)
        dts[8] = copy.deepcopy(dtree_or)
        SER(0,dts[8], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red= cl_no_red ,leaf_loss_quantify=True, leaf_loss_threshold = 0.2,coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)    
        dts[9] = copy.deepcopy(dtree_or)
        SER(0,dts[9], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red= cl_no_red ,ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.9,coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)    
           
        dts[10] = copy.deepcopy(dtree_or)
        SER(0,dts[10], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red= cl_no_red , no_ext_on_cl=True, cl_no_ext= cl_no_red )
        dts[11] = copy.deepcopy(dtree_or)
        SER(0,dts[11], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red= cl_no_red ,no_ext_on_cl=True, cl_no_ext= cl_no_red ,ext_cond=True)  
        
        dts[12] = copy.deepcopy(dtree_or)
        SER(0,dts[12], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red= cl_no_red ,no_ext_on_cl=True, cl_no_ext= cl_no_red ,ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.5,coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)    
        dts[0] = copy.deepcopy(dtree_or)
        SER(0,dts[0], X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red= cl_no_red ,no_ext_on_cl=True, cl_no_ext= cl_no_red ,ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.9,coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin)    
    
        #dt = rfs[1].estimators_[0]
        #dt_no_red = rfs[2].estimators_[0]
        def true_pos(clf,X,y):
            return sum(clf.predict(X[y==1]) == 1)/sum(y==1)
        def false_pos(clf,X,y):
            return sum(clf.predict(X[y==0]) == 1)/sum(y==0)
        
        netoile = 12
        print('score ser:', dts[1].score(X_target_095,y_target_095))
        print('score ser no red:', dts[2].score(X_target_095,y_target_095))
        print('score ser no ext:', dts[3].score(X_target_095,y_target_095))
        print('score ser *:', dts[netoile].score(X_target_095,y_target_095))

        print('tpr ser:', true_pos(dts[1],X_target_095,y_target_095))
        print('tpr ser no red:', true_pos(dts[2],X_target_095,y_target_095))
        print('tpr ser no ext:', true_pos(dts[3],X_target_095,y_target_095))
        print('tpr ser *:', true_pos(dts[netoile],X_target_095,y_target_095))

        print('fpr ser:', false_pos(dts[1],X_target_095,y_target_095))
        print('fpr ser no red:', false_pos(dts[2],X_target_095,y_target_095))
        print('fpr ser no ext:', false_pos(dts[3],X_target_095,y_target_095))
        print('fpr ser *:', false_pos(dts[netoile],X_target_095,y_target_095))
        
        print('nb feuilles ser :',sum(dts[1].tree_.feature == -2))
        print('nb feuilles ser no red :',sum(dts[2].tree_.feature == -2))
    else:
        
        N_EST = 3

        rf_or = RandomForestClassifier(n_estimators = N_EST,max_depth=MAX )
        rf_or.fit(X_source,y_source)
    
        rfs = np.zeros(13,dtype=object)
        
        rfs[1] = SER_RF(rf_or, X_target_005, y_target_005, original_ser=True )    
    
        rfs[2] = SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True,cl_no_red=[1] )
        rfs[3] = SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext=[1] )    
    
        #no ext vars
        rfs[4] = SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext=[1],ext_cond=True)
        rfs[5] = SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext=[1],leaf_loss_quantify=True, leaf_loss_threshold = 0.2)    
        rfs[6] = SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_ext_on_cl=True, cl_no_ext=[1],ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.9)    
        
        #no red vars
        rfs[7] = SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1],ext_cond=True)
        rfs[8] = SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1],leaf_loss_quantify=True, leaf_loss_threshold = 0.2)    
        rfs[9] = SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1],ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.9)    
           
        rfs[10] = SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1], no_ext_on_cl=True, cl_no_ext=[1])
        rfs[11] = SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1],no_ext_on_cl=True, cl_no_ext=[1],ext_cond=True)  
        
        rfs[12] = SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1],no_ext_on_cl=True, cl_no_ext=[1],ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.1)    
        rfs[0] = SER_RF(rf_or, X_target_005, y_target_005, original_ser=False, no_red_on_cl=True, cl_no_red=[1],no_ext_on_cl=True, cl_no_ext=[1],ext_cond=True,leaf_loss_quantify=True, leaf_loss_threshold = 0.9)    
    
        dt = rfs[1].estimators_[0]
        dt_no_red = rfs[2].estimators_[0]
        
        print('score ser:', rfs[1].score(X_target_095,y_target_095))
        print('score ser no red:', rfs[2].score(X_target_095,y_target_095))
        print('score ser no ext:', rfs[3].score(X_target_095,y_target_095))
        print('score ser *:', rfs[0].score(X_target_095,y_target_095))
        
        print('nb feuilles ser :',sum(dt.tree_.feature == -2))
        print('nb feuilles ser no red :',sum(dt_no_red.tree_.feature == -2))
    
    