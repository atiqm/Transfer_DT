# -*- coding: utf-8 -*-
"""
@author: sergio
"""

import numpy as np
from sklearn import tree
import copy


def add_to_parents(dTree, node, values):
    # dic = dTree.tree_.__getstate__().copy()
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
    #print('div index',index)
    else:    
        index = np.argmax(measures_IG)
    
    
    #print('no div index',np.argmax(measures_IG))
    #print('------')
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

    # print("\nNODE ", node_index)
    # print("parents : ", find_parent(tree, node_index))
    # print("prune_lone_instance ", prune_lone_instance)
    # print("value of source at node_index : ", tree.value[node_index, :,
    # :].astype(int))
    tree.weighted_n_node_samples[node_index] = Y_target_node.size
    tree.impurity[node_index] = GINI(current_class_distribution)
    tree.n_node_samples[node_index] = Y_target_node.size
    # print("feat ", phi)
    # print("threshold ", threshold)
    # print("target_sahpe : ", X_target_node.shape)
    # print("val des target pour la feat : ", X_target_node[:, phi])
    # print("maj class before update values", np.argmax(tree.value[node_index]))
    # print("classes ", classes)
    # print("current_class_distribution_source ",
          # current_class_distribution_source)
    # print("current_class_distribution ", current_class_distribution)
    # print("current_class_distribution_noupdate ",
          # current_class_distribution_noupdate)

    # If it is a leaf one, exit
    if tree.children_left[node_index] == -1 and tree.children_right[node_index] == -1:
        # print("it's a leaf")
        # update tree.value
        tree.value[node_index] = current_class_distribution
        return 0

    is_reached_update = (current_class_distribution.sum() != 0)
    is_reached_noupdate = (current_class_distribution_noupdate.sum() != 0)

    is_instance_cl_no_prune = np.sum(tree.value[node_index, :,
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
        Q_source_l, Q_source_r = get_children_distributions(decisiontree,
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
            
        Q_source_parent = get_node_distribution(decisiontree,
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
    import graphviz
    import matplotlib.pyplot as plt
    # Build a dataset
    dataset_length = 100
    D = 2
    X = np.random.randn(dataset_length, D) * 0.1
    X[0:dataset_length // 2, 0] += 0.1
    X[0:dataset_length // 2, 0] += 0.2
    Y = np.ones(dataset_length)
    Y[0:dataset_length // 2] *= 0
    # Train a Tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    # Plot the classification threshold
    # plt.plot(X[0:dataset_length // 2, 0], X[0:dataset_length // 2, 1], "ro")
    # plt.plot(X[dataset_length // 2:, 0], X[dataset_length // 2:, 1], "bo")
    # for node, feature in enumerate(clf.tree_.feature):
    # if feature == 0:
    # plt.axvline(x=clf.tree_.threshold[node])
    # elif feature == 1:
    # plt.axhline(y=clf.tree_.threshold[node])
    # plt.show()
    # plot the tree
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=["feature_" +
                                                   str(i) for i in range(D)],
                                    class_names=["class_0", "class_1"],
                                    filled=True, rounded=True,
                                    special_characters=True)

    graph = graphviz.Source(dot_data)
    graph.render('prarent_tree.gv', view=True)
    # Build a new dataset
    X = np.random.randn(dataset_length, D + 1) * 0.1
    #X[:,1] = np.nan
    X[0:dataset_length // 2, 0] += 0.3
    Y = np.ones(dataset_length)
    Y[0:dataset_length // 2] *= 0
    # Call STRUT
    print("Applying STRUT")

    STRUT(clf, 0, X, Y, 1)
    # Plot the new thresholds

    # plt.plot(X[0:dataset_length // 2, 0], X[0:dataset_length // 2, 1], "ro")
    # plt.plot(X[dataset_length // 2:, 0], X[dataset_length // 2:, 1], "bo")
    # for node, feature in enumerate(clf.tree_.feature):
    # if feature == 0:
    # plt.axvline(x=clf.tree_.threshold[node])
    # elif feature == 1:
    # plt.axhline(y=clf.tree_.threshold[node])
    # plt.show()
    # Plot the new tree
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=["feature_" +
                                                   str(i) for i in range(D)],
                                    class_names=["class_0", "class_1"],
                                    filled=True, rounded=True,
                                    special_characters=True)

    graph = graphviz.Source(dot_data)
    graph.render('child_tree.gv', view=True)
