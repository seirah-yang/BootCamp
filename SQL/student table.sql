use test;

create table alpacostudent(
sid int not null,
sname varchar(10)not null  -- variable character (가변문자)
);

insert into alpacostudent(sid, sname) values(1, "정철우");
insert into alpacostudent(sid, sname) values(2, "김창용");

-- 조회한 다음
select sid, sname
from alpacostudent;

-- 학생 정철우를 제거하자 
set SQL_SAFE_UPDATES = 0; -- 세이프모드를 안쓰겠다. 

delete 
from alpacostudent 
where sname like '정철우'; -- 문자열이라서 '==' 보다는 'like' 추천

select *  -- columns
from alpacostudent;