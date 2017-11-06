import pandas as pd
import sqlite3

import numpy as np
import matplotlib as plt
import S000_setup as s000

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

    #df_match_attr_wide.to_csv('df_match_attr_wide.csv')
    #df_home_pos.to_csv('df_home_pos.csv')
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

    close(conn)
