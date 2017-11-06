import pandas as pd
import numpy as np
import sqlite3
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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

def connect(sqlite_file):
    """ Make connection to an SQLite database file """
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    return conn, c

def close(conn):
    """ Commit changes and close connection to the database """
    # conn.commit()
    conn.close()

def total_rows(cursor, table_name, print_out=False):
    """ Returns the total number of rows in the database """
    c.execute('SELECT COUNT(*) FROM {}'.format(table_name))
    count = c.fetchall()
    if print_out:
        print('\t Total rows: {}'.format(count[0][0]))
    return count[0][0]

def table_col_info(cursor, table_name, print_out=False):
    """ Returns a list of tuples with column informations:
        (id, name, type, notnull, default_value, primary_key) """
    c.execute('PRAGMA TABLE_INFO({})'.format(table_name))
    info = c.fetchall()

    if print_out:
        print("\t  Column Info: ID, Name, Type, NotNull, DefaultVal, PrimaryKey")
        for col in info:
            print("\t  ", col)
    return info

def values_in_col(cursor, table_name, print_out=True):
    """ Returns a dictionary with columns as keys and 
        the number of not-null entries as associated values. """
    c.execute('PRAGMA TABLE_INFO({})'.format(table_name))
    info = c.fetchall()
    col_dict = dict()
    for col in info:
        col_dict[col[1]] = 0
    for col in col_dict:
        c.execute('SELECT ({0}) FROM {1} WHERE {0} IS NOT NULL'.format(col, table_name))
        # In my case this approach resulted in a better performance than using COUNT
        number_rows = len(c.fetchall())
        col_dict[col] = number_rows
    if print_out:
        print("\nNumber of entries per column:")
        for i in col_dict.items():
            print('{}: {}'.format(i[0], i[1]))
    return col_dict

def getTableAsOfMatchDay(league_id, season, match_date, team_api_id):
    """ Returns league table as of match day for home and away teams """
    sql_table_as_of_day = "select league_id, season, team_api_id, count(*) as stage, " \
          "\"Latest\" as Table_Type, sum(GP) as GP, sum(W) as W, sum(L) as L, sum(D) as D, " \
          "sum(GF) as GF, sum(GA) as GA, sum(GD) as GD, sum(PTS) as PTS from (" \
          "select league_id, season, home_team_api_id as team_api_id, " \
          "team_short_name, \"Home\" as Table_Type, count(a.id) as GP, " \
          "sum(case match_result when \"Home_Win\" then 1 else 0 end) as W," \
          "sum(case match_result when \"Away_Win\" then 1 else 0 end) as L," \
          "sum(case match_result when \"Draw\" then 1 else 0 end) as D," \
          "sum(home_team_goal) as GF, sum(away_team_goal) as GA, " \
          "sum(home_team_goal) - sum(away_team_goal) as GD, " \
          "sum(home_team_points) as PTS " \
          "from match_attributes a left join team_master t on a.home_team_api_id = t.team_api_id " \
          "where a.league_id = ? and a.season = ? and a.match_date < ? " \
          "group by a.league_id, a.season, a.home_team_api_id, t.team_short_name " \
          "union " \
          "select league_id, season, away_team_api_id as team_api_id, " \
          "team_short_name, \"Away\" as Table_Type, count(a.id) as GP, " \
          "sum(case match_result when \"Away_Win\" then 1 else 0 end) as W, " \
          "sum(case match_result when \"Home_Win\" then 1 else 0 end) as L, " \
          "sum(case match_result when \"Draw\" then 1 else 0 end) as D, " \
          "sum(away_team_goal) as GF, sum(home_team_goal) as GA, " \
          "sum(away_team_goal) - sum(home_team_goal) as GD, " \
          "sum(away_team_points) as PTS " \
          "from match_attributes a left join team_master t on a.away_team_api_id = t.team_api_id " \
          "where a.league_id = ? and a.season = ? and a.match_date < ? " \
          "group by a.league_id, a.season, a.away_team_api_id, t.team_short_name) " \
          "group by league_id, season, team_api_id " \
          "order by league_id, season, PTS DESC, GD, GF"

    args_table_as_of_day = [league_id, season, match_date, league_id, season, match_date]
    df_table_as_of_day = pd.read_sql_query(sql_table_as_of_day, conn, params=args_table_as_of_day)

    dict_team_pos = {}
    for index, aTeam in df_table_as_of_day.iterrows():
        if aTeam[2] == team_api_id:
            dict_team_pos = aTeam.to_dict()
            dict_team_pos.update({'pos': index})
        else:
            continue
    #print(team_api_id, index, dict_team_pos)
    return dict_team_pos

def getAllTeamAttr():
    """ Returns the team attribute """
    sql_all_team_attr = "select * from team_attributes"
    df_all_team_attr = pd.read_sql_query(sql_all_team_attr, conn)
    df_all_team_attr = df_all_team_attr.drop(['buildUpPlaySpeedClass','buildUpPlayDribblingClass',
                                     'buildUpPlayPassingClass','buildUpPlayPositioningClass',
                                     'chanceCreationPassingClass', 'chanceCreationPositioningClass',
                                     'chanceCreationCrossingClass', 'chanceCreationShootingClass',
                                     'defencePressureClass','defenceAggressionClass',
                                     'defenceTeamWidthClass','defenceDefenderLineClass',
                                     'id', 'team_fifa_api_id', 'attr_date'], axis=1)
    return df_all_team_attr

if __name__ == '__main__':

    sqlite_file = 'eurosoccerdb.sqlite'
    conn, c = connect(sqlite_file)
    sql_match_attr = 'select m.*, h.team_short_name as home_team_short_name, a.team_short_name as away_team_short_name' \
                     ' from match_attributes m' \
                     '  left join team_master h on m.home_team_api_id = h.team_api_id ' \
                     '  left join team_master a on m.away_team_api_id = a.team_api_id ' \
                     'where m.country_id IN (1729, 4769, 7809, 10257, 21518) ' \
                     'order by m.season, m.match_date'
    df_match_attr = pd.read_sql_query(sql_match_attr, conn)
    df_match_attr_wide = pd.DataFrame()

    list_home_pos = []
    list_away_pos = []

    df_match_attr = df_match_attr.drop(['home_player_X1', 'home_player_X2', 'home_player_X3',
                                        'home_player_X4',
                                        'home_player_X5', 'home_player_X6', 'home_player_X7',
                                        'home_player_X8', 'home_player_X9', 'home_player_X10',
                                        'home_player_X11','home_player_Y1', 'home_player_Y2',
                                        'home_player_Y3', 'home_player_Y4', 'home_player_Y5',
                                        'home_player_Y6', 'home_player_Y7', 'home_player_Y8',
                                        'home_player_Y9', 'home_player_Y10', 'home_player_Y11',
                                        'away_player_X1', 'away_player_X2', 'away_player_X3',
                                        'away_player_X4', 'away_player_X5', 'away_player_X6',
                                        'away_player_X7', 'away_player_X8', 'away_player_X9',
                                        'away_player_X10', 'away_player_X11', 'away_player_Y1',
                                        'away_player_Y2', 'away_player_Y3', 'away_player_Y4',
                                        'away_player_Y5', 'away_player_Y6', 'away_player_Y7',
                                        'away_player_Y8', 'away_player_Y9', 'away_player_Y10',
                                        'away_player_Y11',
                                        'goal', 'shoton', 'shotoff', 'foulcommit', 'card',
                                        'cross', 'corner', 'possession'], axis=1)
    for index, aMatch in df_match_attr.iterrows():
        league_id = aMatch['league_id']
        season = aMatch['season']
        match_date = aMatch['match_date']
        home_team_api_id = aMatch['home_team_api_id']
        away_team_api_id = aMatch['away_team_api_id']

        dict_home_pos = {}
        dict_away_pos = {}

        dict_home_pos = getTableAsOfMatchDay(league_id, season, match_date, home_team_api_id)
        list_home_pos.append(dict_home_pos)
        dict_away_pos = getTableAsOfMatchDay(league_id, season, match_date, away_team_api_id)
        list_away_pos.append(dict_away_pos)

    df_home_pos = pd.DataFrame(list_home_pos)
    df_away_pos = pd.DataFrame(list_away_pos)

    df_home_pos.columns = ['home_pos_' + col for col in df_home_pos.columns]
    df_away_pos.columns = ['away_pos_' + col for col in df_away_pos.columns]

    df_home_team_attr = getAllTeamAttr()
    df_away_team_attr = getAllTeamAttr()

    df_home_team_attr.columns = ['home_attr_' + col for col in df_home_team_attr.columns]
    df_away_team_attr.columns = ['away_attr_' + col  for col in df_away_team_attr.columns]

    print('Existing match attributes', df_match_attr.shape)
    print('Adding home team attributes', df_home_team_attr.shape)
    df_match_attr_wide = df_match_attr.merge(df_home_team_attr, how="left",
                                             left_on=['home_team_api_id', 'season'],
                                             right_on=['home_attr_team_api_id', 'home_attr_season'])

    print('Existing match attributes', df_match_attr_wide.shape)
    print('Adding away team attributes', df_away_team_attr.shape)
    df_match_attr_wide = df_match_attr_wide.merge(df_away_team_attr, how="left",
                                                  left_on=['away_team_api_id', 'season'],
                                                  right_on=['away_attr_team_api_id', 'away_attr_season'])

    print('Existing match attributes', df_match_attr_wide.shape)
    print('Adding home team position', df_home_pos.shape)
    df_match_attr_wide = df_match_attr_wide.merge(df_home_pos, how="left",
                                                  left_on=['home_team_api_id', 'season', 'league_id', 'stage'],
                                                  right_on=['home_pos_team_api_id', 'home_pos_season', 'home_pos_league_id', 'home_pos_stage'])

    print('Existing match attributes', df_match_attr_wide.shape)
    print('Adding away team position', df_away_pos.shape)
    df_match_attr_wide = df_match_attr_wide.merge(df_away_pos, how="left",
                                                  left_on=['away_team_api_id', 'season', 'league_id', 'stage'],
                                                  right_on=['away_pos_team_api_id', 'away_pos_season', 'away_pos_league_id', 'away_pos_stage'])

    print('Final data frame', df_match_attr_wide.shape)
    print('Final data frame has columns', df_match_attr_wide.columns)

    df_match_attr_wide = df_match_attr_wide.drop(['home_attr_team_api_id', 'home_attr_season',
                                                  'away_attr_team_api_id', 'away_attr_season',
                                                  'home_pos_Table_Type', 'home_pos_team_api_id',
                                                  'away_pos_Table_Type', 'away_pos_team_api_id',
                                                  'home_pos_season', 'away_pos_season',
                                                  'home_pos_league_id', 'away_pos_league_id',
                                                  'home_pos_stage', 'away_pos_stage'
                                                  ], axis=1)
    # print(df_match_attr_wide.head)
    # df_match_attr_wide.to_csv('test.csv')
    df_match_attr_wide.to_sql(con=conn, name='Match_Attributes_Wide', if_exists='replace')

    #    close(conn)
    #sqlite_file = 'eurosoccerdb.sqlite'
    #conn, c = connect(sqlite_file)
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

    y=df_match_raw.pop('match_result_y')
    X=df_match_raw

    print(type(X), X.shape)
    print(type(y), y.shape)

    # X_team_attr = X.drop('home_pos_W_PCT', 'home_pos_D_PCT', 'home_pos_L_PCT', 'home_pos_GF_PG',
    #    'home_pos_GA_PG', 'home_pos_GD_PG', 'home_pos_PTS_PG', 'away_pos_W_PCT',
    #    'away_pos_D_PCT', 'away_pos_L_PCT', 'away_pos_GF_PG', 'away_pos_GA_PG',
    #    'away_pos_GD_PG', 'away_pos_PTS_PG')

    X_team_pos = X.drop('home_attr_buildUpPlaySpeed',
       'home_attr_buildUpPlayPassing', 'home_attr_chanceCreationPassing',
       'home_attr_chanceCreationCrossing', 'home_attr_chanceCreationShooting',
       'home_attr_defencePressure', 'home_attr_defenceAggression',
       'home_attr_defenceTeamWidth', 'away_attr_buildUpPlaySpeed',
       'away_attr_buildUpPlayPassing', 'away_attr_chanceCreationPassing',
       'away_attr_chanceCreationCrossing', 'away_attr_chanceCreationShooting',
       'away_attr_defencePressure', 'away_attr_defenceAggression',
       'away_attr_defenceTeamWidth')

    # X.plot(kind='box', subplots=True, layout=(5, 8), sharex=False, sharey=False)
    # plt.show()
    # X.hist()
    # plt.show()
    scatter_matrix(X_team_pos)
    plt.show()

    # knn = KNeighborsClassifier()
    # knn.fit(X_train, y_train)
    # KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')
    # knn.predict(X_test)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    # evaluate each model in turn
    results = []
    names = []
    num_folds = 7
    num_instances = len(X_train)
    seed = 7
    scoring = 'accuracy'

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    # Make predictions on validation dataset
    # knn = KNeighborsClassifier()
    # knn.fit(X_train, y_train)
    # predictions = knn.predict(X_test)
    # print(accuracy_score(y_test, predictions))
    # print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))

    # # Make predictions on validation dataset
    # svc = SVC()
    # svc.fit(X_train, y_train)
    # predictions = svc.predict(X_test)
    # print(accuracy_score(y_test, predictions))
    # print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))
