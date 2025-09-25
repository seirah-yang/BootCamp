use test;
select*
from player; 

set SQL_SAFE_UPDATES=0; -- delete/ update 가능하게 safe mode off 
SET AUTOCOMMIT=0; -- 자동저장 off 
commit; -- save 

delete from player where player_name="유동우";

select * from player; -- 유동우가 사라졌다.

rollback; -- 마지막 commit 시점으로 돌아감 
select * from player; -- 유동우가 살아났다. 

-- =====================================
select * from player; -- 정호곤 / 최경훈 
commit; 

delete from player where player_name="정호곤";
savepoint A; -- 정호곤이 사망한 시점 

delete from player where player_name="최경훈";
savepoint b; -- 최경훈이 사망한 시점 

rollback to A;

select * from player; -- 최경훈 살아났음.

-- ============================================
select * 
from player 
where weight between 70 and 90; -- 70이상 90이하 

select *
from player 
where position in ("DF", "FW"); -- DF나 FW가 포지션인 

select *
from player
where e_player_name is null;-- eplayer_name이 null인 

select *
from player
where not e_player_name is null;-- eplayer_name이 null인 
 
-- ====================================================

select player_name, height, CASE
								when height >= 180
									then '장신'
								when height >= 170
									then '일반'
								when height >=160
									then '단신'
								ELSE '기타'
								END AS 기준
from player;

select team_id, count(player_name), sum(weight), 
				avg(weight), max(weight), min(weight), 
                variance(weight), std(weight)
from player -- 개별정보를 group by 할때 넣으면 안됨 
group by team_id

having count(player_name) >= 10; -- 선수인원이 10명 이상인 팀만 

-- ====실행순서 : FORM, WHERE, GROPU BY, HAVING, SELECT, ORDER BY====
-- == 아래 적는 순서는 지켜줘야 함 
select team_id, count(team_id), avg(weight)
from player 
where weight  >= 70
group by team_id
having count(player_id) > 10 -- 집계단위 조건  
order by count(player_id) desc, avg(weight) asc -- 이중 정렬 방법 (default = asc)
limit 3; -- 상위 3 (head 3)

-- =======문제 
select team_id, count(player_id), avg(height)
								from player
								where height >=180
								group by team_id
								having count(player_id) > 5 
								order by count(player_id) desc, avg(height) asc
limit 2;
-- ====================================================
-- 이중조인 
select p.player_name, t.team_name -- 선수테이블에서 선수이름, 팀 테이블에서 팀이름 
from player as p, team as t -- 선수테이블을 p, 팀테이블을 t라고 엘리아스 
where p.team_id = t.team_id; -- 엮는 기준 

-- 3중 조인 
select p.player_name, t.team_name, s.stadium_name 
from player as p, team as t, stadium as s
where p.team_id = t.team_id -- 엮는 기준
and t.stadium_id = s.stadium_id; 


-- player 기준 조인. 
select t.team_id, t.team_name, count(p.player_id)
from player as p 
				left join 
						team as t
	on p.team_id = t.team_id 
where p.weight >=70 -- 몸무게 70이상만 팀마다 집계 
group by team_id
order by count(p.player_id) desc; -- 선수 수로 내림차순 

select * 
from 
		A inner join B
        on A.userId = B.userId;

select*
from 
		A left join B 
        on A.pstId = B.postId;


-- ==================================================
-- 1. where 절에서 서브쿼리 
select *
from player        -- n개가 서브쿼리에서 추출되기 떄문에 동등 연산자하면 안되고 
where player_name in (select player_name
					  from player 
                      where player_name like"김_")
and height >= 180;

-- 2. from 절에서 서브쿼리 
-- 쿼리의 결과를 테이블로 쓰는 것.
select temp.player_name, temp.weight, temp.height, t.team_name
from (select player_name, team_id, weight, height  
	  from player 
      where player_name like "김"
      and height >= 180) temp, team as t 
where temp.team_id = t.team_id;