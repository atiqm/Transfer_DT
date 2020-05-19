import numpy as np


# def depth_rf(rf):
    # d = 0
    # for p in rf.estimators_:
        # d = d + p.tree_.max_depth
    # return d / len(rf.estimators_)


def depth(dtree, node):
    p, t, b = extract_rule(dtree, node)
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


def extract_rule(dtree, node):

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
            node, b = find_parent(dtree, node)

        feats.pop(0)
        ths.pop(0)
        bools.pop(0)
        nodes.pop(0)

    return np.array(feats), np.array(ths), np.array(bools)


def extract_leaves_rules(dtree):
    leaves = np.where(dtree.tree_.feature == -2)[0]

    rules = np.zeros(leaves.size, dtype=object)
    for k, f in enumerate(leaves):
        rules[k] = extract_rule(dtree, f)

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

#    if depth_vtree(tree1, f) + dic2['max_depth'] > dic['max_depth']:
#        dic['max_depth'] = depth_vtree(tree1, f) + tree2.max_depth

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
    if depth(dTree1, f) + dTree2.tree_.max_depth > dTree1.tree_.max_depth:
        dTree1.tree_.max_depth = depth(dTree1, f) + dTree2.tree_.max_depth

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

    dic_old = dic.copy()
    left_old = dic_old['nodes']['left_child']
    right_old = dic_old['nodes']['right_child']

    dic['nodes'] = dic['nodes'][inds]
    dic['values'] = dic['values'][inds]

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

    dic_old = dic.copy()
    left_old = dic_old['nodes']['left_child']
    right_old = dic_old['nodes']['right_child']

    dic['nodes'] = dic['nodes'][inds]
    dic['values'] = dic['values'][inds]

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


def GINI(class_distribution):
    if class_distribution.sum():
        p = class_distribution / class_distribution.sum()
        return 1 - (p**2).sum()
    return 0
