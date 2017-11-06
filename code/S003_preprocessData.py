import pandas as pd
import sqlite3
import S001_exploreSoccerDB_v2 as s001

import matplotlib.pyplot as plt
import numpy as np

print(df_match_raw.shape)

print(df_match_raw.isnull().sum(), df_match_raw.dtypes)
