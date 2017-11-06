
create table League_Table as
select league_id, season, home_team_api_id as team_api_id, team_short_name, "Home" as Table_Type, 
	count(a.id) as GP, 
	sum(case match_result when "Home_Win" then 1 else 0 end) as W,
	sum(case match_result when "Away_Win" then 1 else 0 end) as L,
	sum(case match_result when "Draw" then 1 else 0 end) as D,
	sum(home_team_goal) as GF, sum(away_team_goal) as GA, 
	sum(home_team_goal) - sum(away_team_goal) as GD, 
	sum(home_team_points) as PTS
from match_attributes a left join team_master t on a.home_team_api_id = t.team_api_id
group by a.league_id, a.season, a.home_team_api_id, t.team_short_name
union
select league_id, season, away_team_api_id as team_api_id, team_short_name, "Away" as Table_Type, 
	count(a.id) as GP, 
	sum(case match_result when "Away_Win" then 1 else 0 end) as W,
	sum(case match_result when "Home_Win" then 1 else 0 end) as L,
	sum(case match_result when "Draw" then 1 else 0 end) as D,
	sum(away_team_goal) as GF, sum(home_team_goal) as GA, 
	sum(away_team_goal) - sum(home_team_goal) as GD, 
	sum(away_team_points) as PTS
from match_attributes a left join team_master t on a.away_team_api_id = t.team_api_id
group by a.league_id, a.season, a.away_team_api_id, t.team_short_name


insert into League_Table
select league_id, season, team_api_id, team_short_name, "Final" as Table_Type, 
	sum(GP) as GP, sum(W) as W, sum(L) as L, sum(D) as D, 
	sum(GF) as GF, sum(GA) as GA, sum(GD) as GD, sum(PTS) as PTS
from League_Table
group by league_id, season, team_api_id, team_short_name
order by league_id, season, PTS