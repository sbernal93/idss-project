import numpy as np


def check_children(index, is_leaves, children_right, children_left, values):
    if is_leaves[index]:
        if values[index][0][0]>values[index][0][1]:
            return "no"
        else:
            return "yes"
    else:
        result_right = check_children(children_right[index], is_leaves, children_right, children_left, values)
        result_left  = check_children(children_left[index], is_leaves, children_right, children_left, values)
        if result_right == result_left and result_right != "not-equal":
            return result_right
        else:
            return "not-equal"


def ask_questions(dt, is_leaves, features):
    n_nodes = dt.tree_.node_count
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    values = dt.tree_.value
    binary_features = ['job', 'marital', 'education', 'poutcome', 'default', 'housing', 'loan', 'personal']
    i=0
    answers = {}
    while True:
        if is_leaves[i]:
            if values[i][0][0]>values[i][0][1]:
                result = "no"
            else:
                result = "yes"
            break
        else:
            result = check_children(i, is_leaves, children_right, children_left, values)
            if result != "not-equal":
                break
        if features[feature[i]] in answers:
            answer = answers[features[feature[i]]]
        else:
            feature_found = features[feature[i]]
            if isinstance(feature_found, str) and feature_found.split('_')[0] in binary_features:
                answer = input("Has %s %s? (y/n): " % (feature_found.split('_')[0], feature_found.split('_')[1]))
                if answer == 'yes':
                    answer = 1
                else:
                    answer = 0
            else:
                answer = input("What is the value for %s: " % (feature_found))
            answers[features[feature[i]]] = answer
        # print(answer)
        if float(answer) <= float(threshold[i]):
            i=children_left[i]
        else:
            i=children_right[i]

    print("Result is: %s" % result)


def get_leaves(dt):
    n_nodes = dt.tree_.node_count
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right

    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    return is_leaves
