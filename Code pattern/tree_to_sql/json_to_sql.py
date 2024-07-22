from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
import json

# Генерация произвольного датасета
X, y = make_classification(n_samples=100, n_features=5, n_informative=2, n_clusters_per_class=2, random_state=42)

# Обучение решающего дерева
tree_classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_classifier.fit(X, y)

# Функция для конвертации дерева в JSON
def convert_tree_to_json(tree: DecisionTreeClassifier) -> str:
    """tree to json function"""
    _tree = tree.tree_


    def recurse(node):
        """recurse function to check node is leaf or not"""
        is_leaf = _tree.children_left[node] == -1 and _tree.children_right[node] == -1
        if is_leaf:
            target = int(_tree.value[node].argmax())
            return {"class": target}
        else:
            name = int(_tree.feature[node])
            threshold = round(float(_tree.threshold[node]), 4)
            left = recurse(_tree.children_left[node])
            right = recurse(_tree.children_right[node])
            return {"feature_index": name, "threshold": threshold, "left": left, "right": right}

    tree_json = recurse(0)
    return json.dumps(tree_json, indent=2)


def generate_sql_query(tree_as_json: str, features: list) -> str:
    tree_dict = json.loads(tree_as_json)
    def sql_condition(node, feature_names):
        if "class" in node:
            return f"{node['class']}"
        else:
            feature_name = feature_names[node["feature_index"]]
            threshold = node["threshold"]
            left_condition = sql_condition(node["left"], feature_names)
            right_condition = sql_condition(node["right"], feature_names)
            return f"CASE WHEN {feature_name} > {threshold} THEN {right_condition} ELSE {left_condition} END"

    condition = sql_condition(tree_dict, features)
    sql_query = f"SELECT {condition} AS class_label"
    return sql_query
