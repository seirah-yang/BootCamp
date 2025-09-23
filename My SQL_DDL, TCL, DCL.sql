drop tables alpacolecture, alpacostudent; 
 
use test;
select *
from player;

-- 학원강의 
create table alpacolecture(
lecture_no int primary key auto_increment, 
lecture_name varchar(20) not null
);
-- 학원학생 
create table alpacostudent(
sid int primary key auto_increment, 
sname varchar(20) not null, 
email varchar(20) not null,
lecture_no int not null,
constraint alpa_FK foreign key (lecture_no) references alpacolecture(lecture_no)
);

insert into alpacolecture (lecture_name) values ("마스터과정");

insert into alpacostudent (sname, email,lecture_no) values ("스파이", "test@naver.com", 2);
insert into alpacostudent (sname, email,lecture_no) values ("홍길동", "test@naver.com", 1);

select * from alpacolecture; 

-- 컬럼추가 
alter table alpacolecture add lecture_tile int;

-- 컬럼삭제 
alter table alpacolecture drop column lecture_time;

-- 컬럼타입 수정변경 
alter table alpacolecture modify lecture_name varchar(100);

-- 컬럼명 변경 
alter table alpacolecture rename column lecture_name to lec_name;


select * from player;
-- 김태호의 position을 FW로 바꿔보자 
update player 
set POSITION = "FW" 
where player_id = "2000001";

select * from player;
delete from player
where player_id = "2000001";

select * from player;

-- 선수의 키와 몸무게 출력 
select player_name, weight, height
from player;

select player_name, weight as "몸무게", height as "키"
from player;

-- 선수의 키와 몸무게를 더하고 "몸무게와키"로 컬럼명 출력 
select player_name, weight+height as "몸무게와키"
from player;

select player_name, concat(height,"cm");

select player_name 
from player
where height >= 180
or weight >= 80;

select player_name 
from player
where height >=180
and weight >= 80;

-- 이름이 김민성인 사람 출력
select player_name 
from player
where player_name = "김민성";

select player_name
from player
where player_name like "김_"; -- 성이 김씨이면서 이름이 외자인 경우만 포함 

-- 포지션의 유니크값 출력하기 
select distinct position 
from player;


select player_name
from player
where player_name like "김%"; -- 성이 김씨인 사람 전부 포함 

-- like에서만 쓸 수 있는 퍼센트는 글자 수에 연연하지 않는 와일드 카드임 

-- 중간이름이 '민'이 들어가는 글자 단, %는 0이 될 수도 있따. 
select player_name
from player
where player_name like "%민%";
 