#!/usr/bin/env python

from __future__ import print_function

import pandas as pd
import numpy as np
import traceback
import sys


def replace_nans(df):
    # replace Nan with mean
    avg_rating_by_driver_mean = np.mean(df["avg_rating_by_driver"])
    avg_rating_of_driver_mean = np.mean(df["avg_rating_of_driver"])
    df["avg_rating_by_driver"].fillna(avg_rating_by_driver_mean, inplace=True)
    df["avg_rating_of_driver"].fillna(avg_rating_of_driver_mean, inplace=True)
    return df


def clean_data(df):
    df = df[~pd.isnull(df).any(axis=1)]
    # creat dummy vars for [city, phone]
    df = pd.get_dummies(df, columns=['city','phone'], drop_first=False)
    # Add dates
    df["last_trip_date"] = pd.to_datetime(df["last_trip_date"])
    # Drop signup_date column
    df.drop(["signup_date"], axis=1, inplace=True)
    return df


def data_preprocess(df):
    try:
        # Replace Nans
        df = replace_nans(df)
        df = clean_data(df)
    except Exception as e:

        # Printing this causes the exception to be in the training job logs, as well.
        trc = traceback.format_exc()
        print('Exception during reading data: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

    return df

    if __name__=='__main__':
        pass
