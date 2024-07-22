from typing import Dict

import pandas as pd
from scipy.sparse import csr_matrix


class UserItemMatrix:
    """useritem_matrix class"""
    def _create_csr_matrix(self) -> csr_matrix:
        """create csr matrix from sales data"""
        data = self._sales_data['qty'].values
        row_ind = [self._user_map[user_id] for user_id in self._sales_data['user_id']]
        col_ind = [self._item_map[item_id] for item_id in self._sales_data['item_id']]
        matrix = csr_matrix((data, (row_ind, col_ind)), shape=(self._user_count, self._item_count))
        return matrix

    def __init__(self, sales_data: pd.DataFrame):
        """Class initialization. You can make necessary
        calculations here.

        Args:
            sales_data (pd.DataFrame): Sales dataset.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36
            ...

        """
        self._sales_data = sales_data.copy()

        self._user_count = sales_data['user_id'].nunique()
        self._item_count = sales_data['item_id'].nunique()

        self._user_map = {user_id: idx for idx, user_id in
                          enumerate(sorted(sales_data['user_id'].unique()))}
        self._item_map = {item_id: idx for idx, item_id in
                          enumerate(sorted(sales_data['item_id'].unique()))}

        self._matrix = self._create_csr_matrix()

    #def _create_csr_matrix(self) -> csr_matrix:
    #    """create csr matrix from sales data"""
    #    data = self._sales_data['qty'].values
    #    row_ind = [self._user_map[user_id] for user_id in self._sales_data['user_id']]
    #    col_ind = [self._item_map[item_id] for item_id in self._sales_data['item_id']]
    #    matrix = csr_matrix((data, (row_ind, col_ind)), shape=(self._user_count, self._item_count))
    #    return matrix

    @property
    def user_count(self) -> int:
        """
        Returns:
            int: the number of users in sales_data.
        """
        return self._user_count

    @property
    def item_count(self) -> int:
        """
        Returns:
            int: the number of items in sales_data.
        """
        return self._item_count

    @property
    def user_map(self) -> Dict[int, int]:
        """Creates a mapping from user_id to matrix rows indexes.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36

            user_map (Dict[int, int]):
                {1: 0, 2: 1, 4: 2, 5: 3}

        Returns:
            Dict[int, int]: User map
        """
        return self._user_map

    @property
    def item_map(self) -> Dict[int, int]:
        """Creates a mapping from item_id to matrix rows indexes.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36

            item_map (Dict[int, int]):
                {118: 0, 285: 1, 1229: 2, 1688: 3, 2068: 4}

        Returns:
            Dict[int, int]: Item map
        """
        return self._item_map

    @property
    def csr_matrix(self) -> csr_matrix:
        """User items matrix in form of CSR matrix.

        User row_ind, col_ind as
        rows and cols indecies(mapped from user/item map).

        Returns:
            csr_matrix: CSR matrix
        """
        return self._matrix
