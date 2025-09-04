use test;

drop table content;

create table content (
c_id int primary key auto_increment,
c_title varchar(20) not null, 
c_text varchar(50) not null, 
user_id varchar(20) not null,
date varchar(20) not null
);