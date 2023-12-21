
import numpy as np
import statistics

def supervised_fitness_fn_moo(objectives):
    """
    Returns a fitness function for multi-objective optimization problems.

    Args:
    - objectives: a list of objective functions to optimize

    Returns:
    - fitness_fn: a function that takes a pipeline, a dataset (X, y), and optional arguments,
                  and returns a tuple of scores for each objective function
    """

    def fitness_fn(
        pipeline,
        X,
        y,
        *args,
        validation_split=0.3,
        cross_validation_steps=3,
        cross_validation="median",
        **kwargs
    ):
        """
        Performs cross-validation to evaluate the performance of a pipeline on a dataset.

        Args:
        - pipeline: the pipeline to evaluate
        - X: the input data
        - y: the output data
        - validation_split: the proportion of data to use for validation
        - cross_validation_steps: the number of times to perform cross-validation
        - cross_validation: the function to use to aggregate the cross-validation scores (either 'mean' or 'median')
        - kwargs: additional arguments to pass to the pipeline

        Returns:
        - r_scores: a tuple of scores for each objective function, aggregated over the cross-validation steps
        """

        scores = []
        for _ in range(cross_validation_steps):
            # Split the data into training and validation sets
            len_x = len(X) if isinstance(X, list) else X.shape[0]
            indices = np.arange(0, len_x)
            np.random.shuffle(indices)
            split_index = int(validation_split * len(indices))
            train_indices = indices[:-split_index]
            test_indices = indices[-split_index:]

            # Split the data into training and validation sets
            if isinstance(X, list):
                X_train, y_train, X_test, y_test = (
                    [X[i] for i in train_indices],
                    y[train_indices],
                    [X[i] for i in test_indices],
                    y[test_indices],
                )
            else:
                X_train, y_train, X_test, y_test = (
                    X[train_indices],
                    y[train_indices],
                    X[test_indices],
                    y[test_indices],
                )

            # Train the pipeline on the training set
            pipeline.send("train")
            pipeline.run(X_train, y_train, **kwargs)

            # Evaluate the pipeline on the validation set
            pipeline.send("eval")
            y_pred = pipeline.run(X_test, None, **kwargs)

            # Calculate the scores for each objective function
            scores.append([objective(y_test, y_pred) for objective in objectives])

        # Aggregate the scores over the cross-validation steps
        scores_per_objective = list(zip(*scores))
        r_scores = tuple(
            [
                getattr(statistics, cross_validation)(score)
                for score in scores_per_objective
            ]
        )
        return r_scores

    return fitness_fn


def unsupervised_fitness_fn_moo(objectives):
    """
    Returns a fitness function for unsupervised multi-objective optimization.

    Parameters:
    -----------
    objectives : list
        A list of objective functions to evaluate the performance of the unsupervised model.

    Returns:
    --------
    fitness_fn : function
        A fitness function that takes a pipeline and data X as inputs, runs the pipeline on X,
        and returns the tuple of objective scores.

    Example:
    --------
    >>> from sklearn.cluster import KMeans
    >>> pipeline = KMeans(n_clusters=2)
    >>> objectives = [silhouette_score, calinski_harabasz_score]
    >>> fitness_function = unsupervised_fitness_fn_moo(objectives)
    >>> scores = fitness_function(pipeline, X)
    """

    def fitness_fn(pipeline, X, *args, **kwargs):
        """
        Evaluates the performance of an unsupervised model using the given objectives.

        Parameters:
        -----------
        pipeline : object
            An unsupervised learning model that implements the fit and predict methods.

        X : array-like, shape (n_samples, n_features)
            The input data to train the model.

        Returns:
        --------
        tuple
            A tuple of objective scores.

        """
        pipeline.send("train")
        pipeline.run(X)
        pipeline.send("eval")
        y_pred = pipeline.run(X)
        return tuple([objective(X, y_pred) for objective in objectives])

    return fitness_fn

def accuracy(ytrue, ypred) -> float:
    return np.mean([1 if yt == yp else 0 for yt, yp in zip(ytrue, ypred)])