import pandas as pd
import numpy as np
import sqlite3
import S001_exploreSoccerDB_v2 as s001
import S000_setup as s000
import matplotlib as plt
import matplotlib.pyplot as pyplt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn import model_selection

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

sqlite_file = 'eurosoccerdb.sqlite'
conn, c = s000.connect(sqlite_file)
sql_match_raw = 'select *, case match_result when \'Home_Win\' then 1 when \'Draw\' then 2 else 3 end as match_result_y ' \
                'from Match_Attributes_Wide ' \
                'where home_attr_buildUpPlaySpeed is not null ' \
                'and home_attr_buildUpPlayDribbling is not null ' \
                'and home_attr_buildUpPlayPassing is not null ' \
                'and home_attr_chanceCreationPassing is not null ' \
                'and home_attr_chanceCreationCrossing is not null ' \
                'and home_attr_chanceCreationShooting is not null and ' \
                'home_attr_defencePressure is not null ' \
                'and home_attr_defenceAggression is not null ' \
                'and home_attr_defenceTeamWidth is not null ' \
                'and away_attr_buildUpPlaySpeed is not null ' \
                'and away_attr_buildUpPlayDribbling is not null ' \
                'and away_attr_buildUpPlayPassing is not null ' \
                'and away_attr_chanceCreationPassing is not null ' \
                'and away_attr_chanceCreationCrossing is not null ' \
                'and away_attr_chanceCreationShooting is not null ' \
                'and away_attr_defencePressure is not null ' \
                'and away_attr_defenceAggression is not null ' \
                'and away_attr_defenceTeamWidth is not null ' \
                'and home_pos_D is not null ' \
                'and home_pos_GA is not null ' \
                'and home_pos_GD is not null ' \
                'and home_pos_GF is not null ' \
                'and home_pos_GP is not null ' \
                'and home_pos_L is not null ' \
                'and home_pos_PTS is not null ' \
                'and home_pos_W is not null ' \
                'and home_pos_pos is not null ' \
                'and away_pos_D is not null ' \
                'and away_pos_GA is not null ' \
                'and away_pos_GD is not null ' \
                'and away_pos_GF is not null ' \
                'and away_pos_GP is not null ' \
                'and away_pos_L is not null ' \
                'and away_pos_PTS is not null ' \
                'and away_pos_W is not null ' \
                'and away_pos_pos is not null'

df_match_raw = pd.read_sql_query(sql_match_raw, conn)

df_match_raw['home_pos_W_PCT'] = df_match_raw['home_pos_W'] / df_match_raw['home_pos_GP']
df_match_raw['home_pos_D_PCT'] = df_match_raw['home_pos_D'] / df_match_raw['home_pos_GP']
df_match_raw['home_pos_L_PCT'] = df_match_raw['home_pos_L'] / df_match_raw['home_pos_GP']
df_match_raw['home_pos_GF_PG'] = df_match_raw['home_pos_GF'] / df_match_raw['home_pos_GP']
df_match_raw['home_pos_GA_PG'] = df_match_raw['home_pos_GA'] / df_match_raw['home_pos_GP']
df_match_raw['home_pos_GD_PG'] = df_match_raw['home_pos_GD'] / df_match_raw['home_pos_GP']
df_match_raw['home_pos_PTS_PG'] = df_match_raw['home_pos_PTS'] / df_match_raw['home_pos_GP']

df_match_raw['away_pos_W_PCT'] = df_match_raw['away_pos_W'] / df_match_raw['away_pos_GP']
df_match_raw['away_pos_D_PCT'] = df_match_raw['away_pos_D'] / df_match_raw['away_pos_GP']
df_match_raw['away_pos_L_PCT'] = df_match_raw['away_pos_L'] / df_match_raw['away_pos_GP']
df_match_raw['away_pos_GF_PG'] = df_match_raw['away_pos_GF'] / df_match_raw['away_pos_GP']
df_match_raw['away_pos_GA_PG'] = df_match_raw['away_pos_GA'] / df_match_raw['away_pos_GP']
df_match_raw['away_pos_GD_PG'] = df_match_raw['away_pos_GD'] / df_match_raw['away_pos_GP']
df_match_raw['away_pos_PTS_PG'] = df_match_raw['away_pos_PTS'] / df_match_raw['away_pos_GP']

df_match_raw = df_match_raw.drop(['home_player_1', 'home_player_2', 'home_player_3',
                                  'home_player_4', 'home_player_5', 'home_player_6',
                                  'home_player_7', 'home_player_8', 'home_player_9',
                                  'home_player_10', 'home_player_11',
                                  'away_player_1', 'away_player_2', 'away_player_3',
                                  'away_player_4', 'away_player_5', 'away_player_6',
                                  'away_player_7', 'away_player_8', 'away_player_9',
                                  'away_player_10', 'away_player_11',
                                  'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH',
                                  'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA',
                                  'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD',
                                  'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA',
                                  'match_date', 'match_result', 'season',
                                  'home_team_short_name', 'away_team_short_name',
                                  'home_team_points', 'away_team_points', 'stage',
                                  'home_attr_buildUpPlayDribbling', 'away_attr_buildUpPlayDribbling',
                                  'home_pos_D', 'home_pos_GA', 'home_pos_GD',
                                  'home_pos_GF', 'home_pos_GP', 'home_pos_L', 'home_pos_PTS',
                                  'home_pos_W', 'home_pos_pos', 'away_pos_D', 'away_pos_GA',
                                  'away_pos_GD', 'away_pos_GF', 'away_pos_GP', 'away_pos_L',
                                  'away_pos_PTS', 'away_pos_W'
                                  ], axis=1)

print(df_match_raw.shape)
print(df_match_raw.columns)

# print(df_match_raw.head(10).to_string())
# print(df_match_raw.describe().to_string())

# for column_name in df_match_raw.columns:
#     if column_name in ['index', 'id', 'country_id', 'league_id', 'season',
#                        'match_api_id', 'home_team_api_id', 'away_team_api_id',
#                        'home_team_goal', 'away_team_goal', 'match_result']: continue
#     print('Creating histogram for: ', column_name)
#     column_values = df_match_raw[column_name]
#     n, bins, patches = plt.hist(column_values, 50, normed=1, facecolor='green', alpha=0.75)
#     y = mlab.normpdf( bins, column_values.mean(), column_values.std())
#     if "attr" in column_name:
#         figure = plt.figure()
#         ax1 = figure.add_subplot(121)
#         plt.hist(column_values, facecolor='red', alpha=0.75)
#         #l = plt.plot(bins, y, 'r--', linewidth=1)
#         plt.xlabel(column_name)
#         plt.ylabel('Probability')
#         plt.title('Histogram of %s | mu=%.2f; sigma=%.2f' % (column_name, column_values.mean(), column_values.std()))
#         # plt.axis([40, 160, 0, 0.03])
#         plt.grid(True)
#         plt.show()
#     else:
#         l = plt.plot(bins, y, 'r--', linewidth=1)
#         plt.xlabel(column_name)
#         plt.ylabel('Probability')
#         plt.title('Histogram of %s | mu=%.2f; sigma=%.2f' % (column_name, column_values.mean(), column_values.std()))
#         # plt.axis([40, 160, 0, 0.03])
#         plt.grid(True)
#         plt.show()


y = df_match_raw.pop('match_result_y')
X = df_match_raw
#print(type(X), X.shape)
#print(type(y), y.shape)

X_team_attr = X.drop(['index', 'id', 'country_id', 'league_id', 'match_api_id',
                      'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal',
                      'home_pos_W_PCT', 'home_pos_D_PCT', 'home_pos_L_PCT', 'home_pos_GF_PG',
                      'home_pos_GA_PG', 'home_pos_GD_PG', 'home_pos_PTS_PG', 'away_pos_W_PCT',
                      'away_pos_D_PCT', 'away_pos_L_PCT', 'away_pos_GF_PG', 'away_pos_GA_PG',
                      'away_pos_GD_PG', 'away_pos_PTS_PG'], axis=1)

# s000.plotDataFrame(X_team_attr)

X_team_pos = X.drop(['index', 'id', 'country_id', 'league_id', 'match_api_id',
                     'home_team_api_id', 'away_team_api_id', 'home_team_goal',
                     'away_team_goal',
                     'home_attr_buildUpPlaySpeed',
                     'home_attr_buildUpPlayPassing', 'home_attr_chanceCreationPassing',
                     'home_attr_chanceCreationCrossing', 'home_attr_chanceCreationShooting',
                     'home_attr_defencePressure', 'home_attr_defenceAggression',
                     'home_attr_defenceTeamWidth', 'away_attr_buildUpPlaySpeed',
                     'away_attr_buildUpPlayPassing', 'away_attr_chanceCreationPassing',
                     'away_attr_chanceCreationCrossing', 'away_attr_chanceCreationShooting',
                     'away_attr_defencePressure', 'away_attr_defenceAggression',
                     'away_attr_defenceTeamWidth'], axis=1)

# X_team_pos.plot(kind='box', subplots=True, layout=(5, 8), sharex=False, sharey=False)
# s000.plotDataFrame(X_team_pos)

# test_sizes = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
test_sizes = (0.30)

for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # Make predictions on validation dataset
    # knn = KNeighborsClassifier()
    # knn.fit(X_train, y_train)
    # predictions = knn.predict(X_test)
    # print('\n\ntest size: ', test_size)
    # print(accuracy_score(y_test, predictions))
    # print(confusion_matrix(y_test, predictions))
    # print(test_size, classification_report(y_test, predictions))

    # Make predictions on validation dataset
    # lr = LogisticRegression()
    # lr.fit(X_train, y_train)
    # predictions = lr.predict(X_test)
    # print('\n\ntest size: ', test_size)
    # print(accuracy_score(y_test, predictions))
    # print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))

    # Make predictions on validation dataset
    # svc = SVC()
    # svc.fit(X_train, y_train)
    # predictions = svc.predict(X_test)
    # print('\n\ntest size: ', test_size)
    # print(accuracy_score(y_test, predictions))
    # print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))
    #
    # evaluate each model in turn
    results = []
    names = []
    num_folds = 7
    num_instances = len(X_train)
    seed = 7
    scoring = 'accuracy'
    #
    # Spot Check Algorithms
    models = []
    # models.append(('LR', LogisticRegression()))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC()))
    # models.append(('LASSO', Lasso()))
    #
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print(type(cv_results))
        print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

    #
    fig = pyplt.figure()
    fig.suptitle('Algorithm Comparison: test size = '+ str(test_size))
    ax = fig.add_subplot(111)
    pyplt.boxplot(results)
    ax.set_xticklabels(names)
    pyplt.show()
    #
