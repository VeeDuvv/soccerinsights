select * from league_master

create table Season_Master as
select league_id, season, min(match_date) as season_start, max(match_date) as season_end
from match_attributes
group by league_id, season
