import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    """Gradient boosting regressor."""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        loss="mse",
        verbose=False,
        subsample_size=0.5,
        replace=False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.subsample_size = subsample_size
        self.replace = replace

        if loss == "mse":
            self.loss = self._mse

        self.trees_ = []
        self.base_pred_ = None

    def _mse(self, y_true, y_pred):
        """Compute the MSE loss and its gradient."""
        loss = np.mean((y_true - y_pred) ** 2)
        grad = y_pred - y_true
        return loss, grad

    def _subsample(self, X, y):
        """Subsample the data."""
        if self.replace:
            indices = np.random.choice(len(X), size=int(self.subsample_size * len(X)), replace=True)
        else:
            indices = np.random.choice(len(X), size=int(self.subsample_size * len(X)), replace=False)
        sub_X = X[indices]
        sub_y = y[indices]
        return sub_X, sub_y

    def fit(self, X, y):
        """Fit the model to the data.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """

        # Initialize base predictions
        self.base_pred_ = y.mean()
        y_pred = np.full(len(y), self.base_pred_)

        # Fit the trees
        for _ in range(self.n_estimators):

            _, grad = self.loss(X, y_pred[sub_X])
            # Subsample the data
            sub_X, sub_y = self._subsample(X, y)

            # Compute the gradient
            _, grad = self.loss(sub_y, y_pred[sub_X])

            # Fit a decision tree regressor
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )

            tree.fit(sub_X, -grad)

            # Update the predictions
            y_pred += self.learning_rate * tree.predict(X)

            if self.verbose:
                print(
                    "MSE on train:",
                    round(self.loss(y, y_pred)[0], 3),
                )

            # Store the tree
            self.trees_.append(tree)

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            y: numpy array of shape (n_samples,).
            The predicted values.

        """

        # Initialize the predictions
        predictions = np.full(len(X), self.base_pred_)

        # Compute the predictions of each tree
        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X)

        return predictions
