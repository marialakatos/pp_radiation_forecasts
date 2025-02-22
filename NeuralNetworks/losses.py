
def crps_sample_tf(y_true, y_pred=None):
    """
    Computes the Continuous Ranked Probability Score (CRPS) for a set of predictions and corresponding true observations.
    This implementation is inspired by the `crps_sample()` function from the R `scoringRules` package.

    Args:
        y_true (tf.Tensor): A 2D tensor of shape `[batch_size, 1]` containing the true observed values.
        y_pred (tf.Tensor): A 2D tensor of shape `[batch_size, num_predictions]` containing the predicted values (ensemble forecasts).

    Returns:
        tf.Tensor: A scalar tensor representing the mean Continuous Ranked Probability Score (CRPS) across all observations in the batch.

    Raises:
        ValueError: If `y_pred` is `None` or if the shapes of `y_true` and `y_pred` are incompatible.
                   - `y_true` must have shape `[batch_size, 1]`.
                   - `y_pred` must have shape `[batch_size, num_predictions]`.

    Example:
        >>> y_true = tf.constant([[0.5], [2.0]], dtype=tf.float32)
        >>> y_pred = tf.constant([[0.3, 0.6, 0.8], [1.8, 2.2, 2.4]], dtype=tf.float32)
        >>> crps = crps_sample_tf(y_true, y_pred)
        >>> print(crps)
        tf.Tensor(0.11111115, shape=(), dtype=float32)

    Notes:
        - This function assumes that `y_pred` contains non-negative values.
        - The CRPS is computed for each observation in the batch and then averaged to produce the final result.
    """

    if y_pred is None:
        raise ValueError("y_pred can not be None.")

    # check for shape compatibility
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[1] != 1:
        raise ValueError(
            f"Shape mismatch: Observations (y_true) must have shape [batch_size, 1] and predicted ensemble forecasts (y_pred) must have shape [batch_size, num_predictions]."
            f" Got y_true shape {y_true.shape} and y_pred shape {y_pred.shape}.")

    num_pred_variables = int(y_pred.shape[1])
    y_pred = tf.maximum(y_pred, 0.0)

    c_1n = 1 / num_pred_variables
    x = tf.sort(y_pred, axis=-1, direction='ASCENDING')

    a = np.linspace(start=0.5 * c_1n, stop=1 - 0.5 * c_1n, num=num_pred_variables)
    b = tf.reduce_sum(tf.multiply(tf.subtract(tf.cast(tf.greater(x, y_true), tf.float32), a), tf.subtract(x, y_true)),
                      axis=1)
    crps_results = tf.multiply(b, 2 * c_1n)

    return tf.reduce_mean(crps_results)



def crps_sample(y_true, y_pred=None):
    """
    Computes the Continuous Ranked Probability Score (CRPS) for a set of predictions and corresponding true observations.

    Args:
        y_true (numpy.ndarray): A 1D array of shape `[batch_size]` containing the true observed values.
        y_pred (numpy.ndarray): A 2D array of shape `[batch_size, num_predictions]` containing the predicted values (ensemble forecasts).

    Returns:
        float: The mean Continuous Ranked Probability Score (CRPS) across all observations in the batch.

    Raises:
        ValueError: If `y_pred` is `None` or if the shapes of `y_true` and `y_pred` are incompatible.
                   - `y_true` must have shape `[batch_size]`.
                   - `y_pred` must have shape `[batch_size, num_predictions]`.

    Example:
        >>> y_true = np.array([0.5, 2.0])
        >>> y_pred = np.array([[0.3, 0.6, 0.8], [1.8, 2.2, 2.4]])
        >>> crps = crps_sample(y_true, y_pred)
        >>> print(crps)
        0.11111111111111112

    Notes:
        - This function assumes that `y_pred` contains non-negative values.
        - The CRPS is computed for each observation in the batch and then averaged to produce the final result.
    """

    if y_pred is None:
        raise ValueError("y_pred cannot be None.")

    # Check for shape compatibility
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Shape mismatch: Observations (y_true) must have shape [batch_size] and predicted ensemble forecasts (y_pred) must have shape [batch_size, num_predictions]."
            f" Got y_true shape {y_true.shape} and y_pred shape {y_pred.shape}.")

    num_pred_variables = y_pred.shape[1]

    # Ensure y_pred contains non-negative values
    y_pred = np.maximum(y_pred, 0.0)

    # Calculate constants
    c_1n = 1 / num_pred_variables
    x = np.sort(y_pred, axis=-1)

    # Reshape y_true to be compatible for broadcasting with y_pred
    y_true = y_true.reshape(-1, 1)

    a = np.linspace(0.5 * c_1n, 1 - 0.5 * c_1n, num=num_pred_variables)
    b = np.sum((np.greater(x, y_true) - a) * (x - y_true), axis=1)
    crps_results = np.mean(2 * c_1n * b)

    return crps_results
