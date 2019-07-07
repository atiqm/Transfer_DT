# -*- coding: utf-8 -*-
"""
@author: sergio
"""

import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import copy


def add_to_parents(dTree, node, values):
    p, b = find_parent(dTree.tree_, node)
    if b != 0:
        dTree.tree_.value[p] = dTree.tree_.value[p] + values
        add_to_parents(dTree, p, values)


def get_children_distributions(decisiontree,
                               node_index):
    tree = decisiontree.tree_
    child_l = tree.children_left[node_index]
    child_r = tree.children_right[node_index]
    Q_source_l = tree.value[child_l]
    Q_source_r = tree.value[child_r]
    return [np.asarray(Q_source_l), np.asarray(Q_source_r)]


def get_node_distribution(decisiontree,
                          node_index):
    tree = decisiontree.tree_
    Q = tree.value[node_index]
    return np.asarray(Q)


def compute_class_distribution(classes,
                               class_membership):
    unique, counts = np.unique(class_membership,
                               return_counts=True)
    classes_counts = dict(zip(unique, counts))
    classes_index = dict(zip(classes, range(len(classes))))
    distribution = np.zeros(len(classes))
    for label, count in classes_counts.items():
        class_index = classes_index[label]
        distribution[class_index] = count
    return distribution


def KL_divergence(class_counts_P,
                  class_counts_Q):
    # KL Divergence to assess the difference between two distributions
    # Definition: $D_{KL}(P||Q) = \sum{i} P(i)ln(\frac{P(i)}{Q(i)})$
    # epsilon to avoid division by 0
    epsilon = 1e-8
    class_counts_P += epsilon
    class_counts_Q += epsilon
    P = class_counts_P * 1. / class_counts_P.sum()
    Q = class_counts_Q * 1. / class_counts_Q.sum()
    Dkl = (P * np.log(P * 1. / Q)).sum()
    return Dkl


def H(class_counts):
    # Entropy
    # Definition: $H(P) = \sum{i} -P(i) ln(P(i))$
    epsilon = 1e-8
    class_counts += epsilon
    P = class_counts * 1. / class_counts.sum()
    return - (P * np.log(P)).sum()


def IG(class_counts_parent,
       class_counts_children):
    # Information Gain
    H_parent = H(class_counts_parent)
    H_children = np.asarray([H(class_counts_child)
                             for class_counts_child in class_counts_children])
    N = class_counts_parent.sum()
    p_children = np.asarray([class_counts_child.sum(
    ) * 1. / N for class_counts_child in class_counts_children])
    information_gain = H_parent - (p_children * H_children).sum()
    return information_gain


def JSD(P, Q):
    M = (P + Q) * 1. / 2
    Dkl_PM = KL_divergence(P, M)
    Dkl_QM = KL_divergence(Q, M)
    return (Dkl_PM + Dkl_QM) * 1. / 2


def DG(Q_source_l,
       Q_source_r,
       Q_target_l,
       Q_target_r):
    # compute proportion of instances at left and right
    p_l = Q_target_l.sum()
    p_r = Q_target_r.sum()
    total_counts = p_l + p_r
    p_l /= total_counts
    p_r /= total_counts
    # compute the DG
    return 1. - p_l * JSD(Q_target_l, Q_source_l) - p_r * JSD(Q_target_r, Q_source_r)


def compute_Q_children_target(X_target_node,
                              Y_target_node,
                              phi,
                              threshold,
                              classes):
    # Split parent node target sample using the threshold provided
    # instances <= threshold go to the left
    # instances > threshold go to the right
    decision_l = X_target_node[:, phi] <= threshold
    decision_r = np.logical_not(decision_l)
    Y_target_child_l = Y_target_node[decision_l]
    Y_target_child_r = Y_target_node[decision_r]
    Q_target_l = compute_class_distribution(classes, Y_target_child_l)
    Q_target_r = compute_class_distribution(classes, Y_target_child_r)
    return Q_target_l, Q_target_r


def threshold_selection(Q_source_parent,
                        Q_source_l,
                        Q_source_r,
                        X_target_node,
                        Y_target_node,
                        phi,
                        classes,
                        use_divergence=True,
                        measure_default_IG=True):
    
    #Ne considérer que les valeurs distinctes
    X_phi_sorted = np.array(list(set(X_target_node[:, phi])))
    X_phi_sorted = np.sort(X_phi_sorted)

    nb_tested_thresholds = X_phi_sorted.shape[0] - 1
    
    if nb_tested_thresholds == 0:
        return X_phi_sorted[0]
    
    measures_IG = np.zeros(nb_tested_thresholds)
    measures_DG = np.zeros(nb_tested_thresholds)
    for i in range(nb_tested_thresholds):
        threshold = (X_phi_sorted[i] + X_phi_sorted[i + 1]) * 1. / 2
        Q_target_l, Q_target_r = compute_Q_children_target(X_target_node,
                                                           Y_target_node,
                                                           phi,
                                                           threshold,
                                                           classes)

        measures_IG[i] = IG(Q_source_parent,
                            [Q_target_l, Q_target_r])
        measures_DG[i] = DG(Q_source_l,
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

    else:    
        index = np.argmax(measures_IG)
    

    threshold = (X_phi_sorted[index] + X_phi_sorted[index + 1]) * 1. / 2
    return threshold


def prune_subtree(decisiontree,
                  node_index):
    tree = decisiontree.tree_
    if tree.children_left[node_index] != -1:
        prune_subtree(decisiontree,
                      tree.children_left[node_index])
        tree.children_left[node_index] = -1
    if tree.children_right[node_index] != -1:
        prune_subtree(decisiontree,
                      tree.children_right[node_index])
        tree.children_right[node_index] = -1


def GINI(class_distribution):
    if class_distribution.sum():
        p = class_distribution / class_distribution.sum()
        return 1 - (p**2).sum()
    return 0


def find_parent(tree, i_node):
    p = -1
    b = 0
    dic = tree.__getstate__()
    if i_node != 0 and i_node != -1:
        if i_node in dic['nodes']['left_child']:
            p = list(dic['nodes']['left_child']).index(i_node)
            b = -1
        elif i_node in dic['nodes']['right_child']:
            p = list(dic['nodes']['right_child']).index(i_node)
            b = 1
    return p, b


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
    tree = decisiontree.tree_
    phi = tree.feature[node_index]
    classes = decisiontree.classes_
    threshold = tree.threshold[node_index]
    old_threshold = threshold
    current_class_distribution_source = tree.value[node_index].astype(int)
    current_class_distribution = compute_class_distribution(
        classes, Y_target_node)
    current_class_distribution_noupdate = compute_class_distribution(
        classes, Y_target_node_noupdate)


    tree.weighted_n_node_samples[node_index] = Y_target_node.size
    tree.impurity[node_index] = GINI(current_class_distribution)
    tree.n_node_samples[node_index] = Y_target_node.size


    # If it is a leaf one, exit
    if tree.children_left[node_index] == -1 and tree.children_right[node_index] == -1:
        tree.value[node_index] = current_class_distribution
        return 0

    is_reached_update = (current_class_distribution.sum() != 0)
    is_reached_noupdate = (current_class_distribution_noupdate.sum() != 0)

    is_instance_cl_no_prune = np.sum(tree.value[node_index, :,
                                                cl_no_prune].astype(int))

    # NEW prune_cond
    add_source_value = False

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


    # prune_cond = not is_reached_update or (not pruning_updated_node and (not is_reached_noupdate) and ((not no_prune_on_cl) or (not is_instance_cl_no_prune)))
    # if no target data at all or ((not reached) and (pruning activated or no
    # instance to preserve)), then prune

    if prune_cond:
        # print("PRUNING at node ", node_index)
        prune_subtree(decisiontree,
                      node_index)
        parent_node, b_p = find_parent(tree, node_index)
        # Get the brother index
        if b_p == -1:  # current_node is left_children
            brother_node = tree.children_right[parent_node]
        if b_p == 1:  # current_node is right_children
            brother_node = tree.children_left[parent_node]
        # Get grand parent index
        grand_parent_node, b_gp = find_parent(tree, parent_node)
        # Shunt the parent
        if b_gp == -1:  # parent is left_children of grandparent
            tree.children_left[grand_parent_node] = brother_node

        if b_gp == 1:  # parent is right_children of grandparent
            tree.children_right[grand_parent_node] = brother_node
        # supress the current node
        # tree.children_left[node_index] = -1  # seem useless since already done in prune_subtree func
        # tree.children_right[node_index] = -1
        tree.children_left[parent_node] = -1  # important
        tree.children_right[parent_node] = -1
        return 0

    # else:
        # print("NO Pruning at node ", node_index)
        # return 0

    # update tree.value with target data
    tree.value[node_index] = current_class_distribution
    if add_source_value:
        # print("adding source value to node")
        add_to_parents(decisiontree, node_index,
                       current_class_distribution_source)
        return 0

    # Only one class is present in the node -> terminal leaf
    if (current_class_distribution > 0).sum() == 1:
        # print("Only one class in node {} --> PRUNING".format(node_index))
        prune_subtree(decisiontree,
                      node_index)
        tree.feature[node_index] = -2

        
        return 0


    # update threshold
    if type(threshold) is np.float64:
        Q_source_l, Q_source_r = get_children_distributions(decisiontree,
                                                                node_index)
        Sl = np.sum(Q_source_l)
        Sr = np.sum(Q_source_r)


        if simple_weights:

            Q_source_l = np.multiply(coeffs, Q_source_l)
            Q_source_r = np.multiply(coeffs, Q_source_r)
            

        if adapt_prop:
            Sl = np.sum(Q_source_l)
            Sr = np.sum(Q_source_r)
            Slt = Y_target_node.size
            Srt = Y_target_node.size
            
            D = np.sum(np.multiply(coeffs, Q_source_l))
            Q_source_l = (Slt/Sl)*np.multiply(coeffs,np.divide(Q_source_l,D))
            D = np.sum(np.multiply(coeffs, Q_source_r))
            Q_source_r = (Srt/Sr)*np.multiply(coeffs,np.divide(Q_source_r,D))            
   
            
        Q_source_parent = get_node_distribution(decisiontree,
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
        Q_target_l, Q_target_r = compute_Q_children_target(X_target_node,
                                                           Y_target_node,
                                                           phi,
                                                           t1,
                                                           classes)
        DG_t1 = DG(Q_source_l.copy(),
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


            
        Q_target_l, Q_target_r = compute_Q_children_target(X_target_node,
                                                           Y_target_node,
                                                           phi,
                                                           t2,
                                                           classes)
        DG_t2 = DG(Q_source_r.copy(),
                   Q_source_l.copy(),
                   Q_target_l,
                   Q_target_r)
        

        if DG_t1 >= DG_t2:
            tree.threshold[node_index] = t1
        else:
            tree.threshold[node_index] = t2
            # swap children
            old_child_r_id = tree.children_right[node_index]
            tree.children_right[node_index] = tree.children_left[node_index]
            tree.children_left[node_index] = old_child_r_id

    if tree.children_left[node_index] != -1:
        # Computing target data passing through node NOT updated
        index_X_child_l = X_target_node_noupdate[:, phi] <= old_threshold
        X_target_node_noupdate_l = X_target_node_noupdate[index_X_child_l, :]
        Y_target_node_noupdate_l = Y_target_node_noupdate[index_X_child_l]
        # Computing target data passing through node updated
        threshold = tree.threshold[node_index]
        index_X_child_l = X_target_node[:, phi] <= threshold
        X_target_child_l = X_target_node[index_X_child_l, :]
        Y_target_child_l = Y_target_node[index_X_child_l]

        STRUT(decisiontree,
              tree.children_left[node_index],
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

    if tree.children_right[node_index] != -1:
        # Computing target data passing through node NOT updated
        index_X_child_r = X_target_node_noupdate[:, phi] > old_threshold
        X_target_node_noupdate_r = X_target_node_noupdate[index_X_child_r, :]
        Y_target_node_noupdate_r = Y_target_node_noupdate[index_X_child_r]
        # Computing target data passing through node updated
        threshold = tree.threshold[node_index]
        index_X_child_r = X_target_node[:, phi] > threshold
        X_target_child_r = X_target_node[index_X_child_r, :]
        Y_target_child_r = Y_target_node[index_X_child_r]

        STRUT(decisiontree,
              tree.children_right[node_index],
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
            props_s = get_node_distribution(rf_strut.estimators_[i], 0)
            props_s = props_s / sum(props_s)
            props_t = np.zeros(props_s.size)
            
            for k in range(props_s.size):
                props_t[k] = np.sum(y_target == k) / y_target.size
                
            coeffs = np.divide(props_t, props_s)

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
    
    if solo_tree:

        dtree_or = DecisionTreeClassifier(max_depth=MAX)
        dtree_or.fit(X_source,y_source)
    
        dts = np.zeros(3,dtype=object)

        cl_no_red = [1]
        Nkmin = sum(y_target_005 == cl_no_red )
        root_source_values = get_node_distribution(dtree_or, 0).reshape(-1)

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
    

        def true_pos(clf,X,y):
            return sum(clf.predict(X[y==1]) == 1)/sum(y==1)
        def false_pos(clf,X,y):
            return sum(clf.predict(X[y==0]) == 1)/sum(y==0)
        
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
