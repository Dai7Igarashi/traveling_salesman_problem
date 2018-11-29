# coding: UTF-8

from __future__ import print_function
import numpy as np
import pandas as pd

class Node:
    def __init__(self, data):
        """
        # Structure of data is below.
         data = {
            "g": g,  # Lower limit g in current node.
            "d_index": d_index,  # Index of lowest cost in current table.
            "d_costs": d_costs,  # Lowest cost in current table.
            "table": table  # Routing table of current node.
         }
        """
        self.data = data
        self.left = None
        self.right = None

class SolveSalesman:
    def __init__(self, route):
        self.route = route  # Proposed moving route.
        self.root = self.init_root()  # Root node.
        self.best_path = None  # Current lowest cost path to leaf-node.
        self.lowest_cost = np.inf  # Current lowest cost at leaf-node.
        self.loop_flag = True  # Break loop operation.
        self.finish_count = route.shape[0] - 2  # Condition of finish searching.
        self.total_path = None  # The answer path.

    # Init root node.
    def init_root(self):
        g, d_index = self.calc_Sl_and_Sc(self.route)
        data = {"g": g, "d_index": d_index, "d_costs": 0, "table": self.route}
        return Node(data)

    # Set data in each node.
    def set_data(self, node):
        for path in self.best_path:
            node = getattr(node, path)
        node.left = self.create_left_node(node.data)
        node.right = self.create_right_node(node.data)
        self.best_path = None

    # Create data for left node.
    def create_left_node(self, data):
        d_costs = data["d_costs"] + self.calc_lowest_cost(data["table"], data["d_index"])
        table = self.trim_and_create_table(data["table"], data["d_index"])
        g, d_index = self.calc_Sl_and_Sc(table)
        g += d_costs
        new_data = {"g": g, "d_index": d_index, "d_costs": d_costs, "table": table}
        return Node(new_data)

    # Create data for right node.
    def create_right_node(self, data):
        table = self.set_inf_and_create_table(data["table"], data["d_index"])
        g, d_index = self.calc_Sl_and_Sc(table)
        g += data["d_costs"]
        new_data = {"g": g, "d_index": d_index, "d_costs": data["d_costs"], "table": table}
        return Node(new_data)

    # Search tree and determine best_path and lowest_cost.
    def search_tree(self, node, which, current_best_path):
        if not which is None:
            current_best_path.append(which)
        else:
            self.lowest_cost = np.inf
        if node.left is None:  # if current node has no children:
            if node.data["g"] <= self.lowest_cost:  # Find the leaf-node which has the lowest cost.
                self.lowest_cost = node.data["g"]  # Set the current lowest cost.
                self.best_path = current_best_path.copy()  # Set the current best_path.
                if self.best_path.count("left") >= self.finish_count:
                    self.create_total_path(self.root)  # Create the answer path.
                    self.loop_flag = False
            if len(current_best_path) > 0:
                current_best_path.pop(-1)
        else:
            self.search_tree(node.left, "left", current_best_path)
            self.search_tree(node.right, "right", current_best_path)

    # Calculate the Sl, Sc, and d_index of current table.
    def calc_Sl_and_Sc(self, table):
        min_in_row = np.asarray(table.min(axis=1))  # List of minimum cost in each row.
        s_row = np.sum(min_in_row)  # Correspond to Sl in text.
        new_table = table.copy()
        new_table = new_table.sub(min_in_row, axis=0)
        min_in_col = np.asarray(new_table.min(axis=0))  # List of minimum cost in each col.
        s_col = np.sum(min_in_col)
        d_index = self.search_d_index(table)  # Index of lower cost row/col in D.
        return s_row + s_col, d_index

    def search_d_index(self, table):
        index_list = list(table.index)
        columns_list = list(table.columns)
        min_cost = np.inf
        min_index = None
        min_column = None

        for index in index_list:
            for column in columns_list:
                current_cost = table.loc[index, column]
                if current_cost < min_cost:
                    min_cost = current_cost
                    min_index = index
                    min_column = column
        return (min_index, min_column)

    # Calculate lowest cost d in D.
    def calc_lowest_cost(self, table, d_index):
        return table.loc[d_index[0], d_index[1]]

    # Scale down and create new table.
    def trim_and_create_table(self, table, d_index):
        new_table = table.copy()
        new_table = new_table.drop(d_index[0], axis=0)
        new_table = new_table.drop(d_index[1], axis=1)
        return new_table

    # Set infinite in d_index and create new table.
    def set_inf_and_create_table(self, table, d_index):
        new_table = table.copy()
        new_table.loc[d_index[0], d_index[1]] = np.inf
        return new_table

    # Create the final output path.
    def create_total_path(self, node):
        approved_paths = []
        forbidden_paths = []
        for path in self.best_path:
            node = getattr(node, path)
            if path == "left":
                approved_paths.append(np.asarray(node.data["d_index"]))
            else:
                forbidden_paths.append(np.asarray(node.data["d_index"]))

        print(approved_paths)
        print(forbidden_paths)

    def __call__(self):
        while True:
            self.search_tree(self.root, None, [])
            if not self.loop_flag:
                break
            self.set_data(self.root)


        print("Cost: {}".format(self.lowest_cost))


def main():
    ROUTE1 = np.asarray([
        [np.inf, 21, 7, 13, 15],
        [11, np.inf, 19, 12, 25],
        [15, 24, np.inf, 13, 5],
        [6, 17, 9, np.inf, 22],
        [28, 6, 11, 5, np.inf]
    ])

    ROUTE2 = np.asarray([
        [np.inf, 21, 5, 15, 9],
        [17, np.inf, 12, 6, 24],
        [13, 5, np.inf, 20, 8],
        [9, 12, 7, np.inf, 23],
        [26, 7, 13, 8, np.inf]
    ])

    ROUTE1 = pd.DataFrame(ROUTE1, index=[1,2,3,4,5], columns=[1,2,3,4,5])
    ROUTE2 = pd.DataFrame(ROUTE2, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])

    # Answering the sample route.
    my_salesman = SolveSalesman(ROUTE1)
    my_salesman()

    # Answering the route of quiz.
    my_salesman = SolveSalesman(ROUTE2)
    my_salesman()


if __name__ == '__main__':
    main()