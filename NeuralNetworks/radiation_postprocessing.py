import numpy as np
import pandas as pd
from models import build_hidden_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from custom_callbacks import TerminateOnNaN
from preprocessing import standardize_features_and_validate, load_radiation_data, load_station_info, \
    flatten_training_array, remove_nan_and_validate
from losses import custom_loss_crps_wrapper_fct, crps_sample_tf, crps_sample
from jproperties import Properties
from keras import backend as K
from aux_functions import reshape_distribution_parameters, reshape_corrected_ensemble
import logging
import os

"""
Censored Normal Distributional Regression Network (DRN) / Neural Network for Postprocessing Radiation Forecasts from WRF  
Input features: raw ensemble mean and variance, number of ensemble members predicting zero radiation,  
lead time, and station coordinates (latitude, longitude, and altitude).
"""

# Configure the logger to display INFO-level messages (e.g., verification day and CRPS calculations)
logging.basicConfig(level=logging.INFO)
# Create a logger instance for the current module
logger = logging.getLogger(__name__)

MODEL = "CORRENS"   # Options: DRN (for CN0 distributional regression network) or CORRENS (for corrected ensemble)
config = Properties()

with open('./property_files/config_' + MODEL + '.properties', 'rb') as config_file:
    config.load(config_file)

TRAIN = int(config.get('TRAIN').data)
HIDDEN_NODES = [int(nodes) for nodes in config.get('hidden_nodes').data.split(',')]
N_FEATURES = int(config.get('N_FEATURES').data)
STATION_IDs = [int(nodes) for nodes in config.get('STATION_IDs').data.split(',')]
BATCH_SIZE = int(config.get('BATCH_SIZE').data)
LEARNING_RATE = float(config.get('LEARNING_RATE').data)


N_OUTPUT = 2 if MODEL == "DRN" else 8
END_DAY = 365
N_STATIONS = len(STATION_IDs)
DAYS = [24, 48]
N_LEAD_TIMES = 24
LAG = 2
START_DAY = TRAIN + LAG + 1
N_MODELS = 10
START_DATE = '2021-01-01'
END_DATE = '2021-12-31'
dates = pd.date_range(start=START_DATE, end=END_DATE)
RADIATION_DATA_PATH = "radSumDataFull.RData"    # contains raw forecasts for the first 24 hours, the second 24 hours, and corresponding observations
STATION_INFO_PATH = "info_stats.csv"    # contains station info (e.g., latitude, longitude and elevation)

MAX_NUMBER_OF_EPOCHS = 500
TEST_SIZE = 0.2
SEED = 60

# load raw forecasts and corresponding observations
lat, lon, alt = load_station_info(path=STATION_INFO_PATH)
ensforecasts_day1, ensforecasts_day2, verobs = load_radiation_data(path=RADIATION_DATA_PATH)

# define early stopping callback for training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# initialize N_MODELS different models for sequential training
if MODEL == "DRN":
    models = [build_hidden_model(N_FEATURES, N_OUTPUT, HIDDEN_NODES, compile_model=True,
                                 optimizer='Adam', lr=LEARNING_RATE, loss=custom_loss_crps_wrapper_fct(LB=0))
              for _ in range(N_MODELS)]
else:
    models = [build_hidden_model(N_FEATURES, N_OUTPUT, HIDDEN_NODES, compile_model=True,
                                 optimizer='Adam', lr=LEARNING_RATE, loss=crps_sample_tf)
              for _ in range(N_MODELS)]

# Training and evaluation process
for day in DAYS:
    for d in range(START_DAY - 1, END_DAY):
        # Logs the actual verification day
        logger.info(str(day) + "/" + dates[d].strftime('%Y-%m-%d'))

        train_idx = range(d - TRAIN - (1 if day == 24 else 2), d - (1 if day == 24 else 2))
        ensforecasts = ensforecasts_day1 if day == 24 else ensforecasts_day2

        observations_train = verobs[:, :, train_idx]
        ens_features_train = ensforecasts[:, :, train_idx, ]
        ens_features_test = ensforecasts[:, :, d, :]
        observations_test = verobs[:, :, d]

        X_train, y_train, X_test, y_test = flatten_training_array(ens_train=ens_features_train,
                                                                  obs_train=observations_train,
                                                                  ens_test=ens_features_test,
                                                                  obs_test=observations_test, latitude=lat,
                                                                  longitude=lon, altitude=alt,
                                                                  n_stations=N_STATIONS,
                                                                  n_lead_times=N_LEAD_TIMES,
                                                                  train=TRAIN, n_features=N_FEATURES,
                                                                  stat_ids=STATION_IDs)

        X_train, y_train, X_test, y_test = remove_nan_and_validate(X_train=X_train, y_train=y_train, X_test=X_test,
                                                                   y_test=y_test)
        # Skip the days for which there is no test data
        if X_test.size == 0:
            continue

        # Extract station IDs and lead times for later use
        valid_stats = X_test[:, N_FEATURES]
        leadtimes_test = X_test[:, 1]
        # Remove station IDs from test dataset
        X_test = X_test[:, 0:N_FEATURES]

        # Standardize the data and check whether there are any NaN-s present in the data after that
        X_train_standardized, X_test_standardized = standardize_features_and_validate(X_train=X_train,
                                                                                      X_test=X_test)

        # Create training and validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train_standardized, y_train,
                                                          test_size=TEST_SIZE,
                                                          random_state=SEED)

        # Fit models
        for i, model in enumerate(models):
            history = model.fit(X_train, y_train,
                                epochs=MAX_NUMBER_OF_EPOCHS,
                                batch_size=BATCH_SIZE,
                                validation_data=(X_val, y_val),
                                callbacks=[TerminateOnNaN(), early_stopping],
                                verbose=0)

            # Predict location and scale of the CN0 distribution
            predictions = model.predict(X_test_standardized, verbose=0)

            if MODEL == "DRN":

                loc_test = predictions[:, 0]
                var_test = predictions[:, 1] ** 2

                loc_test = loc_test.reshape(-1, 1)
                var_test = var_test.reshape(-1, 1)
                leadtimes_test = leadtimes_test.reshape(-1, 1)
                stations_test = valid_stats.reshape(-1, 1)

                par_df = pd.DataFrame({
                    'loc_test': loc_test.flatten(),
                    'var_test': var_test.flatten(),
                    'leadtimes_test': leadtimes_test.flatten(),
                    'stations_test': stations_test.flatten(),
                    'obs_test': y_test
                })

                distPars = reshape_distribution_parameters(df=par_df,
                                                           station_ids=STATION_IDs,
                                                           n_stations=N_STATIONS,
                                                           n_lead_times=N_LEAD_TIMES)

                os.makedirs(f'./Results/censored_normal_drn/{MODEL}/iter{i}', exist_ok=True)
                np.save(
                    f'./Results/censored_normal_drn/{MODEL}/iter{i}/distPars_{dates[d].strftime("%Y-%m-%d")}_{day}.npy',
                    distPars)
            else:

                predictions = np.maximum(predictions, 0.0)

                # Save corrected ensemble forecasts
                leadtimes_test = leadtimes_test.reshape(-1, 1)
                stations_test = valid_stats.reshape(-1, 1)

                df = pd.DataFrame({
                    **{f'ens_{i + 1}': predictions[:, i].flatten() for i in range(N_OUTPUT)},
                    'leadtimes_test': leadtimes_test.flatten(),
                    'stations_test': stations_test.flatten(),
                    'obs_test': y_test
                })

                corrEns = reshape_corrected_ensemble(df=df,
                                                     station_ids=STATION_IDs,
                                                     n_stations=N_STATIONS,
                                                     n_lead_times=N_LEAD_TIMES)

                os.makedirs(f'./Results/corrected_ensemble/{MODEL}/iter{i}', exist_ok=True)
                np.save(f'./Results/corrected_ensemble/{MODEL}/iter{i}/corrEns_{dates[d].strftime("%Y-%m-%d")}_{day}.npy', corrEns)

            K.clear_session()
