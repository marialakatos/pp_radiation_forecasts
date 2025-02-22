from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from losses import crps_normal_censored, crps_sample

def reshape_corrected_ensemble(df, station_ids, n_stations, n_lead_times):
    """
        Reshapes a DataFrame of corrected ensemble forecasts into a 3D numpy array.

        Parameters:
        -----------
        **df** : pandas.DataFrame
            DataFrame with ensemble forecasts and observations.

        **station_ids** : list
            List of station IDs.

        **n_stations** : int
            Number of stations.

        **n_lead_times** : int
            Number of lead times.

        Returns:
        --------
        **corrEns** : numpy.ndarray
            3D array of shape (n_stations, n_lead_times, 9) containing ensemble forecasts and observations for the current
            verification day.
    """
    corrEns = np.full((n_stations, n_lead_times, 9), np.nan)

    for i in range(n_stations):
        for j in range(n_lead_times):
            filtered_df = df[(df['stations_test'] == station_ids[i]) & (df['leadtimes_test'] == j)]
            if filtered_df.empty:
                continue

            for k in range(8):
                corrEns[i, j, k] = filtered_df[f"ens_{k+1}"].item()
            corrEns[i, j, 8] = filtered_df["obs_test"].item()

    return corrEns

def reshape_distribution_parameters(df, station_ids, n_stations, n_lead_times):
    """
        Reshapes a DataFrame of CN0 distribution parameters into a 3D numpy array.

        Parameters:
        -----------
        **df** : pandas.DataFrame
            DataFrame with ensemble forecasts and observations.

        **station_ids** : list
            List of station IDs.

        **n_stations** : int
            Number of stations.

        **n_lead_times** : int
            Number of lead times.

        Returns:
        --------
        **distPars** : numpy.ndarray
            3D array of shape (n_stations, n_lead_times, 9) containing distribution parameters and observations for the current
            verification day.
    """
    distPars = np.full((n_stations, n_lead_times, 3), np.nan)

    for i in range(0, n_stations):
        for j in range(0, n_lead_times):
            condition_station = df['stations_test'] == station_ids[i]
            condition_lead_time = df['leadtimes_test'] == j
            filtered_df = df[condition_station & condition_lead_time]
            if filtered_df.size == 0:
                continue
            distPars[i, j, 0] = filtered_df["loc_test"].item()
            distPars[i, j, 1] = filtered_df["var_test"].item()
            distPars[i, j, 2] = filtered_df["obs_test"].item()

    return distPars

def calculate_feature_importance(model, X_test_standardized, y_test, base_crps, network_type, n_features):
    """
    Calculate permutation importance by shuffling each feature and observing the change in CRPS.
    """
    feature_importances = []
    for i in range(n_features):
        X_test_permuted = X_test_standardized.copy()
        np.random.shuffle(X_test_permuted[:, i])
        predictions = model.predict(X_test_permuted, verbose=0)

        if network_type == "DRN":
            loc_test = predictions[:, 0]
            var_test = predictions[:, 1] ** 2
            crps_permuted = crps_normal_censored(y_test, loc=loc_test, var=var_test, lb=0)
            mean_crps_permuted = np.mean(crps_permuted)
            feature_importance = mean_crps_permuted - base_crps
            feature_importances.append(feature_importance)
        else:
            predictions = np.maximum(predictions, 0.0)
            crps_permuted = crps_sample(y_test, predictions)
            mean_crps_permuted = np.mean(crps_permuted)
            feature_importance = mean_crps_permuted - base_crps
            feature_importances.append(feature_importance)

    return np.array(feature_importances)