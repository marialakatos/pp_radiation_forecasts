import pandas as pd
import numpy as np
import rpy2.robjects as robjects

def load_station_info(path):
    """
    Loads station information from a CSV file and extracts latitude, longitude,
    and altitude data for each station.

    Parameters:
    -----------
    path : str
        The file path to the CSV file containing the station information.

    Returns:
    --------
    3 lists:
        - List of latitude values for each station.
        - List of longitude values for each station.
        - List of altitude values for each station.
    """

    stat_info = pd.read_csv(path)
    lat = list(stat_info["lat"])
    lon = list(stat_info["long"])
    alt = [int(item.replace(' Mts.', '')) for item in list(stat_info["alt"])]
    return lat, lon, alt

def load_radiation_data(path):
    """
    Loads radiation ensemble forecasts and observations from an RData file.

    Parameters:
    -----------
    path : str, optional
        The file path to the RData file containing the radiation data.

    Returns:
    --------
    3 arrays:
        - Ensemble forecasts for day 1
        - Ensemble forecasts for day 2
        - Corresponding observations
    """

    # load RData file
    robjects.r['load'](path)
    rdata = robjects.r
    # access R objects by name
    ensforecasts_day1 = np.array(rdata['ensSumDay1'])
    ensforecasts_day2 = np.array(rdata['ensSumDay2'])
    verobs = np.array(rdata['verobs'])

    return ensforecasts_day1, ensforecasts_day2, verobs

def standardize_features_and_validate(X_train, X_test):
    """
    Standardizes the features of the training and testing datasets and validates that no NaN values are introduced
    during the standardization process.

    Parameters:
    -----------
    **X_train** : array-like
        2D array of shape (n_stations * n_lead_times * train, n_features) containing
        training data features.

    **X_test** : array-like
        2D array of shape (n_stations * n_lead_times * n_features, n_features) containing
        testing data features.

    Returns:
    --------
    standardized testing and training data : 2 arrays
       2 arrays containing the standardized versions of the training and testing datasets.
    """

    mean = np.mean(X_train, axis=0)
    sd = np.std(X_train, axis=0)
    sd = [x if x != 0 else 0.000001 for x in sd] # replace small standard deviations with a small value
    X_train_standardized = (X_train - mean) / sd
    X_test_standardized = (X_test - mean) / sd

    assert not np.isnan(X_train_standardized).any(), "NaNs found in standardized training features"
    assert not np.isnan(X_test_standardized).any(), "NaNs found in standardized test features"

    return X_train_standardized, X_test_standardized

def flatten_training_array(ens_train, obs_train, ens_test, obs_test, latitude, longitude, altitude, n_stations, n_lead_times, train, n_features, stat_ids, n_members=8):
    """
    Reshapes the input data arrays from their original multidimensional form into 2D arrays,
    where each row corresponds to a specific station and lead time combination.

    Parameters:
    -----------
    **ens_train** : array-like
        4D array of shape (n_stations, n_lead_times, train, n_features) containing
        training data features.

    **obs_train** : array-like
        3D array of shape (n_stations, n_lead_times, train) containing training data labels.

    **ens_test** : array-like
        3D array of shape (n_stations, n_lead_times, n_features) containing testing data features.

    **obs_test** : array-like
        2D array of shape (n_stations, n_lead_times) containing testing data labels.

    **latitude** : list
        List of length n_stations containing latitude values for each station.

    **longitude** : list
        List of length n_stations containing longitude values for each station.

    **altitude** : list
        List of length n_stations containing altitude values for each station.

    **n_stations** : int
        Number of stations.

    **n_lead_times** : int
        Number of lead times.

    **train** : int
        Length of training period (in days).

    **n_features** : int
        Number of features in the training data.
        **Features: ensemble mean, lead time, latitude, longitude, altitude, ratio of forecasts predicting 0, and ensemble variance.**

    **stat_ids** : list
        List of length n_stations containing IDs for each station.

    **n_members** : list
        Number of ensemble members.

    Returns:
    --------
    Flattened (2D) version of the 4 original arrays.
    """

    # Initialize counters and empty arrays for training and testing data
    count_train = 0
    count_test = 0

    # Create empty arrays to store flattened training and testing data
    X_train_flattened = np.full((n_stations * n_lead_times * train, n_features), np.nan)
    y_train_flattened = np.full((n_stations * n_lead_times * train), np.nan)
    X_test_flattened = np.full((n_stations * n_lead_times, n_features + 1), np.nan)
    y_test_flattened = np.full((n_stations * n_lead_times), np.nan)

    # Loop through each station and lead time to populate the flattened arrays
    for station in range(0, n_stations):
        for leadtime in range(0, n_lead_times):
            # For testing data:
            # Flatten the input features for each station and lead time
            X_test_flattened[count_test, 0] = np.nanmean(
                ens_test[station, leadtime, 0:n_members])  # Ensemble mean (8 members)
            X_test_flattened[count_test, 1] = leadtime  # Lead time (lt)
            X_test_flattened[count_test, 2] = latitude[station]  # Station latitude
            X_test_flattened[count_test, 3] = longitude[station]  # Station longitude
            X_test_flattened[count_test, 4] = altitude[station]  # Station altitude
            X_test_flattened[count_test, 5] = ens_test[station, leadtime, 8]  # Proportion of zero irradiance
            X_test_flattened[count_test, 6] = ens_test[station, leadtime, 9]  # Ensemble variance
            X_test_flattened[count_test, 7] = stat_ids[station]  # Station ID (to be removed later)

            # Flatten the observed values for each station and lead time
            y_test_flattened[count_test, ] = obs_test[station, leadtime]
            count_test += 1

            # For training data:
            for td in range(0, train):
                # Flatten the input features for each training time step
                X_train_flattened[count_train, 0] = np.nanmean(
                    ens_train[station, leadtime, td, 0:n_members])  # Ensemble mean (8 members)
                X_train_flattened[count_train, 1] = leadtime  # Lead time (lt)
                X_train_flattened[count_train, 2] = latitude[station]  # Station latitude
                X_train_flattened[count_train, 3] = longitude[station]  # Station longitude
                X_train_flattened[count_train, 4] = altitude[station]  # Station altitude
                X_train_flattened[count_train, 5] = ens_train[station, leadtime, td, 8]  # Proportion of zero irradiance
                X_train_flattened[count_train, 6] = ens_train[station, leadtime, td, 9]  # Ensemble variance

                # Flatten the observed values for each training time step
                y_train_flattened[count_train, ] = obs_train[station, leadtime, td]
                count_train += 1

    # Return the flattened input and output arrays for both training and testing
    return X_train_flattened, y_train_flattened, X_test_flattened, y_test_flattened

def remove_nan_and_validate(X_train, y_train, X_test, y_test):
    """
    Removes rows containing NaN values from the training and testing datasets and validates
    that no NaNs remain after the removal process.

    Parameters:
    -----------
    **X_train** : array-like
        2D array of shape (n_stations * n_lead_times * train, n_features) containing
        training data features.

    **y_train** : array-like
        2D array of shape (n_stations * n_lead_times * train, n_features) containing
        training data labels.

    **X_test** : array-like
        2D array of shape (n_stations * n_lead_times * n_features, n_features) containing
        testing data features.

    **y_test** : array-like
        2D array of shape (n_stations * n_lead_times, n_features) containing
        testing data labels.

    Raises:
    -------
    AssertionError
        If any NaN values are found in the datasets after the removal process.

    Returns:
    --------
    Cleaned arrays
    """

    # Remove rows with NaNs in training data
    not_nan_X_train_indices = ~np.isnan(X_train).any(axis=1)
    not_nan_y_train_indices = ~np.isnan(y_train)
    not_nan_indices = not_nan_X_train_indices & not_nan_y_train_indices
    X_train = X_train[not_nan_indices]
    y_train = y_train[not_nan_indices]

    # Remove rows with NaNs in testing data
    not_nan_indices_test = (~np.isnan(X_test).any(axis=1)) & (~np.isnan(y_test))
    X_test = X_test[not_nan_indices_test]
    y_test = y_test[not_nan_indices_test]

    # Ensure no NaNs remain after removal
    assert not np.isnan(X_train).any(), "NaNs found in X_train after removal"
    assert not np.isnan(y_train).any(), "NaNs found in y_train after removal"
    assert not np.isnan(X_test).any(), "NaNs found in X_test after removal"
    assert not np.isnan(y_test).any(), "NaNs found in y_test after removal"

    # Return cleaned data
    return X_train, y_train, X_test, y_test
