import pandas as pd
import sqlite3
from pandas.tools.plotting import scatter_matrix
import numpy as np

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


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

def plotDataFrame(df):
    df.plot(kind='box', subplots=True, layout=(5, 8), sharex=False, sharey=False)
    plt.show()
    df.hist()
    plt.show()
    # scatter_matrix(df)
    # plt.show()