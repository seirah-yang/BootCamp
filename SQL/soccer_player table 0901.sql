create database test;
use test;

CREATE TABLE stadium (
STADIUM_ID    CHAR(3) NOT NULL,
STADIUM_NAME  VARCHAR(40) NOT NULL,
HOMETEAM_ID   CHAR(3),
SEAT_COUNT    int,
ADDRESS       VARCHAR(60),
DDD           VARCHAR(3),
TEL           VARCHAR(10),
CONSTRAINT STADIUM_PK PRIMARY  KEY (STADIUM_ID)
);
CREATE TABLE team (
TEAM_ID     CHAR(3) NOT NULL,
REGION_NAME VARCHAR(8) NOT NULL,
TEAM_NAME   VARCHAR(40) NOT NULL,
E_TEAM_NAME VARCHAR(50),
ORIG_YYYY   CHAR(4),
STADIUM_ID  CHAR(3) NOT NULL,
ZIP_CODE1   CHAR(3),
ZIP_CODE2   CHAR(3),
ADDRESS     VARCHAR(80),
DDD         VARCHAR(3),
TEL         VARCHAR(10),
FAX         VARCHAR(10),
HOMEPAGE    VARCHAR(50),
OWNER       VARCHAR(10),
CONSTRAINT TEAM_PK PRIMARY  KEY (TEAM_ID),
CONSTRAINT TEAM_FK FOREIGN KEY (STADIUM_ID) REFERENCES stadium(STADIUM_ID)
);

CREATE TABLE  player (
PLAYER_ID     CHAR(7) NOT NULL,
PLAYER_NAME   VARCHAR(20) NOT NULL,
TEAM_ID       CHAR(3) NOT NULL,
E_PLAYER_NAME VARCHAR(40),
NICKNAME      VARCHAR(30),
JOIN_YYYY     CHAR(4),
POSITION      VARCHAR(10),
BACK_NO       int(2),
NATION        VARCHAR(20),
BIRTH_DATE    DATE,
SOLAR         CHAR(1),
HEIGHT        int(3),
WEIGHT        int(3),
CONSTRAINT PLAYER_PK PRIMARY  KEY (PLAYER_ID),
CONSTRAINT PLAYER_FK FOREIGN KEY (TEAM_ID) REFERENCES team(TEAM_ID)
);

INSERT INTO stadium VALUES ('D03','전주월드컵경기장','K05',28000,'전북 전주시 덕진구 반월동 763-1','063','273-1763');
INSERT INTO stadium VALUES ('B02','성남종합운동장','K08',27000,'경기도 성남시 분당구 야탑동 486','031','753-3956');
INSERT INTO stadium VALUES ('C06','포항스틸야드','K03',25000,'경북 포항시 남구 괴동 1','054','282-2002');
INSERT INTO stadium VALUES ('D01','광양전용경기장','K07',20009,'전남 광양시 금호동 700','061','792-5600');
INSERT INTO stadium VALUES ('B05','서울월드컵경기장','K09',66806,'서울특별시 마포구 난지도길 66','02','2128-2973');
INSERT INTO stadium VALUES ('B01','인천월드컵경기장','K04',35000,'인천광역시 남구 문학동 482','031','666-0496');
INSERT INTO stadium VALUES ('C05','창원종합운동장','K11','27085','경남 창원시 두대동 145','055','6644-8468');
INSERT INTO stadium VALUES ('C04','울산문수경기장','K01',46102,'울산광역시 남구 옥산 산5','052','220-2468');
INSERT INTO stadium VALUES ('D02','대전월드컵경기장','K10',41000,'대전광역시 유성구 노은동 270','042','252-2002');
INSERT INTO stadium VALUES ('B04','수원월드컵경기장','K02',50000,'수원시 팔달구 우만1동 228번지','031','259-2150');
INSERT INTO stadium VALUES ('A02','광주월드컵경기장','K12','40245','광주광역시 서구 풍암동 423-2','062','2468-8642');
INSERT INTO stadium VALUES ('C02','부산아시아드경기장','K06',30000,'부산광역시 서구 서대신동 3가 210번지','051','247-5771');
INSERT INTO stadium VALUES ('A03','강릉종합경기장','K13',33000,'강원 강릉시 교동 산58','033','459-3631');
INSERT INTO stadium VALUES ('A04','제주월드컵경기장','K14',42256,'제주 서귀포시 법환동','064','3631-2460');
INSERT INTO stadium VALUES ('A05','대구월드컵경기장','K15',66422,'대구광역시 수성구 대흥동 504','053','602-2011');
INSERT INTO stadium VALUES ('F01','대구시민경기장','',30000,'대구광역시','054','');
INSERT INTO stadium VALUES ('F02','부산시민경기장','',30000,'부산광역시','051','');
INSERT INTO stadium VALUES ('F03','일산경기장','',20000,'경기도 일산시','031','');
INSERT INTO stadium VALUES ('F04','마산경기장','',20000,'경남 마산시','055','');
INSERT INTO stadium VALUES ('F05','안양경기장','',20000,'경기도 안양시','031','');
commit;

INSERT INTO team VALUES ('K05','전북','현대모터스','CHUNBUK HYUNDAI MOTORS FC','1995','D03','560','190',
'전북 전주시 덕진구 반월동 763-1 전주월드컵경기장 내','063','273-1763','273-1762','http://www.hyundai-motorsfc.com','');
INSERT INTO team VALUES ('K08','성남','일화천마','SEONGNAM ILHWA CHUNMA FC','1988','B02','462',
'130','경기도 성남시 분당구 야탑동 486번지 성남 제2종합운동장 내','031','753-3956','753-4443','http://www.esifc.com','');
INSERT INTO team VALUES ('K03','포항','스틸러스','FC POHANG STEELERS','1973','C06','790','050',
'경북 포항시 죽도동 614-8 동양빌딩 7층','054','282-2002','282-5955','http://www.steelers.co.kr','');
INSERT INTO team VALUES ('K07','전남','드래곤즈','CHUNNAM DRAGONS FC','1994','D01','544','010',
'전남 광양시 중동 1318-5 신한은행빌딩 2층','061','792-5600','792-5605','http://www.dragons.co.kr','');
INSERT INTO team VALUES ('K09','서울','FC서울','FOOTBALL CLUB SEOUL','1983','B05','138','221',
'서울 마포구 성산동 515 월드컵 경기장 내','02','2005-5746','2005-5802','http://www.fcseoul.com','');
INSERT INTO team VALUES ('K04','인천','유나이티드','INCHEON UNITED FC','2004','B01','110','728',
'인천광역시 남구 문학동 482 인천월드컵경기장 내','032','2121-5271','2121-5276','http://www.incheonutd.com','');
INSERT INTO team VALUES ('K11','경남','경남FC','GYEONGNAM FC','2006','C05','111','222',
'경남 창원시 두대동 145 창원종합운동장','055','6644-8468','6644-8488','http://www.gsndfc.co.kr','');
INSERT INTO team VALUES ('K01','울산','울산현대','ULSAN HYUNDAI FC','1986','C04','682','060',
'울산광역시 동구 서부동 산137-1 현대스포츠클럽하우스','052','230-6141','230-6145','http://www.uhfc.tv','');
INSERT INTO team VALUES ('K10','대전','시티즌','DAEJEON CITIZEN FC','1996','D02','301','030',
'대전광역시 유성구 노은동 270 대전월드컵경기장 서관 3층','042','252-2002','221-0669','http://www.dcfc.co.kr','');
INSERT INTO team VALUES ('K02','수원','삼성블루윙즈','SUWON SAMSUNG BLUEWINGS FC','1995','B04','440','220',
'수원시 팔달구 우만1동 228 수원월드컵경기장 4층','031','247-2002','257-0766','http://www.bluewings.kr','');
INSERT INTO team VALUES ('K12','광주','광주상무','GWANGJU SANGMU FC','1984','A02','450','419',
'광주광역시 서구 풍암동 423-2 광주월드컵경기장 내','062','777-5180','777-5181','http://www.gwangjusmfc.co.kr','');
INSERT INTO team VALUES ('K06','부산','아이파크','BUSAN IPARK FC','1983','C02','570','050',
'부산광역시 강서구 대저1동 2155번지 강서체육공원 내','051','555-7101','555-7103','http://www.busanipark.co.kr','');
INSERT INTO team VALUES ('K13','강원','강원FC','GANGWON FC','2008','A03','333','444',
'강원 춘천시 중앙로 7 강원도 개발공사 빌딩 내','033','459-3631','459-3630','http://www.gangwon-fc.com','');
INSERT INTO team VALUES ('K14','제주','제주유나이티드FC','JEJU UNITED FC','1982','A04','555','666',
'제주 서귀포시 강정동 3355-5','064','3631-2460','3631-2468','http://www.jeju-utd.com','');
INSERT INTO team VALUES ('K15','대구','대구FC','DAEGU FC','2002','A05','777','888',
'대구광역시 수성구 대흥동 504 대구스타디움 내','053','5566-3967','5566-3970','http://www.daegufc.co.kr','');
commit;

INSERT INTO player VALUES ('2009175','우르모브','K06','','','2009','DF','4','유고','1987-08-30','1','180','70');
INSERT INTO player VALUES ('2007188','윤희준','K06','','','2005','DF','15','','1982-12-01','1','180','74');
INSERT INTO player VALUES ('2012073','김규호','K06','','','2011','DF','23','','1989-07-13','1','177','72');
INSERT INTO player VALUES ('2007178','김민성','K06','','','','DF','20','','1983-06-23','1','182','73');
INSERT INTO player VALUES ('2007191','김장관','K06','','배추도사,작은삼손','2007','DF','18','','1984-06-05','2','170','61');
INSERT INTO player VALUES ('2008384','김정효','K06','','깜둥이,통키통','2008','DF','19','','1985-07-23','2','174','68');
INSERT INTO player VALUES ('2008395','장대일','K06','','달구','2010','DF','7','','1985-03-09','2','184','79');
INSERT INTO player VALUES ('2011050','박상수','K06','','꼬마홍길동','2011','DF','36','','1988-06-14','1','173','65');
INSERT INTO player VALUES ('2007189','정재영','K06','','제리','2006','MF','6','','1983-09-02','2','187','75');
INSERT INTO player VALUES ('2011049','정태민','K06','','킹카','2011','MF','38','','1992-05-02','1','180','71');
INSERT INTO player VALUES ('2010107','정현우','K06','','','2010','MF','37','','1991-06-04','1','178','70');
INSERT INTO player VALUES ('2011043','송종국','K06','','썰렁왕자','2011','MF','24','','1989-02-20','1','177','73');
INSERT INTO player VALUES ('2011044','오정석','K06','','서경석','2011','MF','13','','1988-09-08','1','178','74');
INSERT INTO player VALUES ('2012137','이고르','K06','이골 실바 데 페리이따스','이골','2012','MF','21','브라질','1989-10-25','1','181','76');
INSERT INTO player VALUES ('2007200','김용하','K06','','용식이','2007','MF','26','','1983-12-15','2','173','66');
INSERT INTO player VALUES ('2012072','전상배','K06','','','2012','MF','14','','1989-03-22','1','176','67');
INSERT INTO player VALUES ('2009038','전우근','K06','','에너자이져','2009','MF','11','','1987-02-25','2','175','68');
INSERT INTO player VALUES ('2008365','이태성','K06','','','2011','MF','30','','1987-06-16','1','177','64');
INSERT INTO player VALUES ('2011047','황철민','K06','','사다리맨','2011','MF','35','','1988-11-20','1','175','68');
INSERT INTO player VALUES ('2008235','정관규','K06','','','','FW','39','','1986-10-10','1','178','71');
INSERT INTO player VALUES ('2011048','정기종','K06','','','2011','FW','25','','1985-05-22','1','182','78');
INSERT INTO player VALUES ('2012074','정창오','K06','','임땡','2012','FW','27','','1988-01-10','1','186','82');
INSERT INTO player VALUES ('2012127','디디','K06','Sebastiao Pereira do Nascimento','','2012','FW','8','브라질','1986-02-24','1','179','78');
INSERT INTO player VALUES ('2007182','마니치','K06','','바람의 아들','2006','FW','9','','1982-01-16','2','184','80');
INSERT INTO player VALUES ('2007195','우성용','K06','','따따','2006','FW','22','','1983-08-18','2','191','76');
INSERT INTO player VALUES ('2010103','장기봉','K06','','짝팔','2010','FW','12','','1987-07-08','1','180','71');
INSERT INTO player VALUES ('2012075','이광수','K06','','','2012','FW','29','','1989-09-25','1','177','73');
INSERT INTO player VALUES ('2010087','하리','K06','','','2010','FW','10','콜롬비아','1984-05-14','1','168','65');
INSERT INTO player VALUES ('2000017','박상남','K06','','','2008','FW','32','','1985-09-07','2','188','80');
INSERT INTO player VALUES ('2000018','빅토르','K06','','빅토르','2011','FW','28','나이지리아','1985-01-05','1','183','79');
INSERT INTO player VALUES ('2000021','이윤겸','K04','LEE, YOONGYUM','','2002','DF' ,'31','','1989-04-20','2','180','72');

INSERT INTO player VALUES ('2000022','하재훈','K04','HA, JAEHON','','2002','DF' ,'31','','1989-04-20','2','180','72');
INSERT INTO player VALUES ('2000023','김충호','K04','KIM, CHUNGHO','','2009','DF' ,'31','','1989-04-20','2','180','72');
INSERT INTO player VALUES ('2000024','임기한','K04','LIM, GIHAN','','2004','DF' ,'31','','1989-04-20','2','180','72');
INSERT INTO player VALUES ('2000025','김경태','K04','KIM, KYOUNGTAE','','','DF','31','','1989-04-20','2','180','72');
																		
INSERT INTO player VALUES ('2012054','남현우','K04','','','','GK','31','','1989-04-20','2','180','72');

INSERT INTO player VALUES ('2008499','김충호','K04','','','','GK','60','','1978-07-04','2','185','83');
INSERT INTO player VALUES ('2011021','이현','K04','','','','GK','1','','1988-11-07','2','192','85');
INSERT INTO player VALUES ('2012052','한동진','K04','','','','GK','21','','1989-08-25','2','183','78');
INSERT INTO player VALUES ('2012126','다오','K04','','','','DF','61','','1992-09-25','2','190','80');
INSERT INTO player VALUES ('2008182','최철','K04','','','','DF','15','','1983-08-20','2','176','77');
INSERT INTO player VALUES ('2010112','송창남','K04','','','','DF','23','','1987-12-31','2','175','67');
INSERT INTO player VALUES ('2008424','조승호','K04','','','','DF','2','','1985-05-13','2','179','70');
INSERT INTO player VALUES ('2008450','윤중희','K04','','','','DF','5','','1985-12-08','1','177','75');
INSERT INTO player VALUES ('2011022','김범직','K04','','','','DF','25','','1989-02-11','1','182','75');
INSERT INTO player VALUES ('2012053','김상홍','K04','','','','DF','30','','1989-02-04','1','178','68');
INSERT INTO player VALUES ('2000001','김태호','K10','','','','DF' ,'31','','1989-04-20','2','180','72');
INSERT INTO player VALUES ('2000002','정상수','K10','JEONG, SAMSOO','','','DF' ,'31','','1989-04-20','2','180','72');
INSERT INTO player VALUES ('2000003','유동우','K10','YOU, DONGWOO','','','DF','40','','1978-03-07','1','177','70');
INSERT INTO player VALUES ('2000004','전기현','K10','JEON, GIHYUN','','','DF' ,'31','','1989-04-20','2','180','72');
INSERT INTO player VALUES ('2012047','강성일','K10','KANG, SUNGIL','','2012','GK','30','','1989-06-04','1','182','80');
INSERT INTO player VALUES ('2010057','김승준','K10','KIM, SEUNG06','개구멍','2010','GK','1','','1982-09-01','1','183','77');
INSERT INTO player VALUES ('2007298','이은성','K10','LEE, EUNSUNG','수호천황','2007','GK','21','','1981-04-05','1','184','82');
INSERT INTO player VALUES ('2007312','정대수','K10','JEONG, DAESOO','','2007','DF','15','','1985-03-20','1','184','74');
INSERT INTO player VALUES ('2012051','정민기','K10','','','','DF','3','','1988-04-25','1','171','65');
INSERT INTO player VALUES ('2010110','정성근','K10','JEONG, SUNGKEUN','','2010','DF','33','','1987-06-20','1','180','73');
INSERT INTO player VALUES ('2011098','정영근','K10','JEONG, YOUNGKWEN','','2011','DF','5','','1988-10-12','1','180','70');
INSERT INTO player VALUES ('2007301','정정수','K10','JEONG, 06GSOO','','2002','DF','36','','1985-01-17','2','180','74');
INSERT INTO player VALUES ('2007309','김창엽','K10','KIM, CHANGYUP','','2007','DF','6','','1984-11-19','2','174','64');
INSERT INTO player VALUES ('2007299','장철우','K10','01G, CHULWOO','폭주기관차','2010','DF','7','','1981-04-01','2','172','65');
INSERT INTO player VALUES ('2011116','콜리','K10','O03 PAPA COLY','검은낙타(Black Camel)','2011','DF','29','세네갈','1985-05-20','1','182','75');
INSERT INTO player VALUES ('2007313','홍광철','K10','HONG, KWANGCHUL','','2007','DF','4','','1984-10-09','2','172','65');
INSERT INTO player VALUES ('2008461','강정훈','K10','KANG, 06GHOON','','2008','MF','38','','1986-02-20','1','175','65');
INSERT INTO player VALUES ('2007306','공오균','K10','KONG, OHKYUN','CROW','2007','MF','22','','1984-08-10','2','177','72');
INSERT INTO player VALUES ('2012049','정국진','K10','JEONG, KOOKJIN','','2012','MF','16','','1988-02-09','1','172','62');
INSERT INTO player VALUES ('2011099','정동선','K10','JEONG, DONGSUN','','2011','MF','9','','1988-03-15','1','170','65');
INSERT INTO player VALUES ('2010109','최경규','K10','CHOI, KUNGGUY','','2010','MF','10','','1987-03-10','1','177','72');
INSERT INTO player VALUES ('2010111','최내철','K10','CHOI, RAECHEOL','','2010','MF','24','','1987-08-20','1','177','71');
INSERT INTO player VALUES ('2012048','배성재','K10','BAE, SUNGJAE','','2012','MF','28','','1989-07-01','1','178','74');
INSERT INTO player VALUES ('2012121','샴','K10','','','','MF','25','','1989-04-30','1','174','69');
INSERT INTO player VALUES ('2012136','오비나','K10','','','','MF','26','','1990-06-03','1','169','70');
INSERT INTO player VALUES ('2010056','김관우','K10','KIM, KWANWOO','SIRIUS','2010','MF','8','','1988-02-25','1','175','69');
INSERT INTO player VALUES ('2012050','김광진','K10','KIM, KWANGJIN','','2012','MF','13','','1982-05-27','1','175','75');
INSERT INTO player VALUES ('2010113','김상규','K10','KIM, SANGKYU','','2010','MF','27','','1987-09-05','1','175','65');
INSERT INTO player VALUES ('2009151','임영주','K10','LIM, YOUNGJOO','','2009','MF','23','','1986-03-08','2','176','68');
INSERT INTO player VALUES ('2011101','박영훈','K10','PARK, YOUNGHOON','','2011','MF','12','','1985-05-01','1','174','73');
INSERT INTO player VALUES ('2008178','한정국','K10','HAN, 06GKOOK','','2011','MF','19','','1981-07-19','1','179','71');
INSERT INTO player VALUES ('2012046','정광선','K10','JEONG, KWANGSUN','','2012','FW','32','','1993-06-17','1','175','68');
INSERT INTO player VALUES ('2007315','정은중','K10','JEONG, EUN06G','샤프(SHARP)','2007','FW','18','','1989-04-08','2','184','72');
INSERT INTO player VALUES ('2009152','정찬중','K10','JEONG. CHANJOONG','','2009','FW','17','','1986-06-14','2','182','72');
INSERT INTO player VALUES ('2011032','김석','K10','KIM, SEOK','','2012','FW','20','','1989-02-01','1','194','85');
INSERT INTO player VALUES ('2011100','탁준석','K10','TAK, 06SUK','','2011','FW','11','','1988-03-24','1','178','69');
INSERT INTO player VALUES ('2000011','정호곤','K06','','','2010','DF' ,'31','','1989-04-20','2','180','72');
INSERT INTO player VALUES ('2000012','최경훈','K06','','','','DF' ,'31','','1989-04-20','2','180','72');
INSERT INTO player VALUES ('2000013','정동훈','K06','','','2010','DF' ,'31','','1989-04-20','2','180','72');
INSERT INTO player VALUES ('2000014','정남표','K06','','','2005','DF' ,'31','','1989-04-20','2','180','72');
INSERT INTO player VALUES ('2000015','정광재','K06','','','2005','DF' ,'31','','1989-04-20','2','180','72');
INSERT INTO player VALUES ('2000016','권혁준','K06','','','2006','DF' ,'31','','1989-04-20','2','180','72');
INSERT INTO player VALUES ('2011052','정경진','K06','','임꺽정','2011','GK','41','','1988-02-07','2','186','78');
INSERT INTO player VALUES ('2012076','정용대','K06','','','2012','GK','40','','1989-10-11','1','189','83');
INSERT INTO player VALUES ('2010108','정지혁','K06','','','2010','GK','31','','1991-11-22','1','187','77');
INSERT INTO player VALUES ('2010059','박유석','K06','','터프가이','2010','GK','1','','1987-06-10','2','186','78');
INSERT INTO player VALUES ('2011053','정진우','K06','','터프가이','2011','DF','33','','1993-02-28','1','180','76');
INSERT INTO player VALUES ('2007185','정학철','K06','','','2005','DF','3','','1982-11-07','2','176','73');
INSERT INTO player VALUES ('2007193','류병훈','K06','','','2005','DF','17','','1986-07-03','2','184','68');
INSERT INTO player VALUES ('2011055','최준홍','K06','','말머리','2011','DF','2','','1988-04-13','1','178','70');
INSERT INTO player VALUES ('2011046','서용혁','K06','','터프가이','2011','DF','34','','1988-07-02','1','183','79');
INSERT INTO player VALUES ('2010058','심재원','K06','','도날드 덕','2010','DF','5','','1987-03-11','1','184','77');
INSERT INTO player VALUES ('2007123','김임생','K04','','','','DF','20','','1981-11-17','1','183','80');
INSERT INTO player VALUES ('2007022','장형석','K04','','','','DF','36','','1982-07-07','1','181','72');
INSERT INTO player VALUES ('2012058','박진성','K04','','','','DF','35','','1993-08-10','1','184','76');
INSERT INTO player VALUES ('2009125','이거룩','K04','','','','DF','4','','1986-06-26','1','183','77');
INSERT INTO player VALUES ('2010001','이정민','K04','','','','DF','3','','1987-10-07','1','183','78');
INSERT INTO player VALUES ('2012134','페르난도','K04','','','','DF','44','','1988-02-24','1','178','74');
INSERT INTO player VALUES ('2000094','김무건','K03','KIM, MUGYUN','','','DF' ,'31','','1989-04-20','2','180','72');
INSERT INTO player VALUES ('2007001','정병지','K03','JEONG, BYUNGJI','','2011','GK','1','','1980-04-08','1','184','77');
INSERT INTO player VALUES ('2011069','최주호','K03','CHOI, JUHO','','2011','GK','51','','1992-07-16','1','185','75');
INSERT INTO player VALUES ('2007130','김대희','K03','KIM, DAEHEE','','2010','GK','31','','1984-04-26','1','192','88');
INSERT INTO player VALUES ('2009133','김준호','K03','KIM, 06HO','','2009','GK','21','','1983-04-28','1','185','77');
INSERT INTO player VALUES ('2011065','허인무','K03','HEO, INMOO','','2011','GK','41','','1988-04-14','1','187','81');
INSERT INTO player VALUES ('2011056','강용','K03','KANG, YONG','','2011','DF','2','','1989-01-14','1','179','72');
INSERT INTO player VALUES ('2007080','고병운','K03','GO, BYUNGWOON','','2006','DF','16','','1983-09-28','1','180','73');
INSERT INTO player VALUES ('2012069','정광석','K03','JEONG, KWANGSUK','','2012','DF','39','','1993-02-12','1','183','72');
INSERT INTO player VALUES ('2007002','정상훈','K03','JEONG, SANGHUN','','2006','DF','13','','1983-06-08','1','183','76');
INSERT INTO player VALUES ('2012062','정석우','K03','JEONG, SEOKWOO','','2012','DF','32','','1993-05-06','1','181','72');
INSERT INTO player VALUES ('2009139','정은석','K03','JEONG, EUNSEOK','','2009','DF','5','','1982-03-14','1','185','80');
INSERT INTO player VALUES ('2009030','최민서','K03','CHOI, MINSEO','','2009','DF','3','','1986-08-24','2','180','78');
INSERT INTO player VALUES ('2012064','성종현','K03','SUNG, JONGHUN','','2012','DF','34','','1989-04-02','1','178','74');
INSERT INTO player VALUES ('2008468','싸빅','K03','Jasenko Sabitovic','','2007','DF','4','','1983-03-29','1','185','78');
INSERT INTO player VALUES ('2007277','오명관','K03','OH, MYUNGKWAN','','2008','DF','15','','1984-04-29','1','185','76');
INSERT INTO player VALUES ('2012063','김동식','K03','KIM, DONGSIK','','2012','MF','33','','1989-03-15','1','183','77');
INSERT INTO player VALUES ('2012067','김수길','K03','KIM, SOOKIL','','2012','DF','37','','1989-04-09','1','179','69');
INSERT INTO player VALUES ('2008444','김승엽','K03','KIM, SEUNGYUB','','2008','DF','12','','1985-10-12','1','180','74');
INSERT INTO player VALUES ('2007101','김종화','K03','','','','DF','25','','1984-04-04','1','182','76');
INSERT INTO player VALUES ('2007190','하석주','K03','HA, SEOKJU','','2011','DF','17','','1978-02-20','1','174','71');
INSERT INTO player VALUES ('2010065','허제정','K03','HEO, JAE06G','','2010','DF','29','','1987-06-02','1','175','70');
INSERT INTO player VALUES ('2007096','홍명보','K03','HONG, MYUNGBO','','2012','DF','20','','1979-02-12','1','181','72');
INSERT INTO player VALUES ('2008472','정기남','K03','JEONG, KINAM','','2010','MF','6','','1981-01-18','1','174','72');
INSERT INTO player VALUES ('2011059','정상록','K03','JEONG, SANGROK','','2011','MF','14','','1989-02-25','1','173','63');
INSERT INTO player VALUES ('2011068','나희근','K03','NA, HEEKEUN','','2011','FW','22','','1989-05-05','1','178','72');
INSERT INTO player VALUES ('2012133','레오','K03','','','','MF','45','','1984-10-22','1','179','74');
INSERT INTO player VALUES ('2012124','메도','K03','Medvid Ivan','','2012','MF','44','','1987-10-13','1','180','78');
INSERT INTO player VALUES ('2012068','최길영','K03','CHOI, KILYOUNG','','2012','MF','38','','1990-02-04','1','172','64');
INSERT INTO player VALUES ('2008203','최상인','K03','CHOI, SANGIN','','2005','DF','27','','1986-03-10','1','174','63');
INSERT INTO player VALUES ('2011131','옐라','K03','Josko Jelicic','','2012','MF','8','','1981-01-05','1','184','79');
INSERT INTO player VALUES ('2011061','유현구','K03','YOU, HYUNGOO','','2011','MF','26','','1993-01-25','1','170','68');
INSERT INTO player VALUES ('2009033','김기부','K03','KIM, KIBU','','2009','MF','18','','1986-03-16','2','183','76');
INSERT INTO player VALUES ('2012070','김상인','K03','KIM, SANGIN','','2012','DF','40','','1989-07-11','1','178','70');
INSERT INTO player VALUES ('2011062','김정운','K03','KIM, 06GWOON','','2011','MF','19','','1988-04-19','1','178','70');
INSERT INTO player VALUES ('2012071','김중규','K03','KIM, 06GJYU','','2012','MF','42','','1989-06-06','1','170','64');
INSERT INTO player VALUES ('2012066','김창호','K03','KIM, CHANGHO','','2012','MF','36','','1991-03-15','1','175','75');
INSERT INTO player VALUES ('2011063','이종범','K03','LEE, JONGBUM','','2011','MF','24','','1988-03-27','1','178','68');
INSERT INTO player VALUES ('2012061','남익경','K03','NAM, IKKYUNG','','2012','MF','30','','1993-01-26','1','177','70');
INSERT INTO player VALUES ('2011064','윤보영','K03','YOON, BOYOUNG','','2011','FW','23','','1988-04-29','1','177','63');
INSERT INTO player VALUES ('2008443','김동국','K03','KIM, DONGGOOK','','2008','FW','10','','1989-04-29','1','185','80');
INSERT INTO player VALUES ('2012060','박종완','K03','PARK, JONGWAN','','2012','DF','28','','1993-08-05','1','188','82');
INSERT INTO player VALUES ('2012059','김징요','K03','Jorge Claudio','','2011','FW','7','브라질','1985-10-01','1','174','70');
INSERT INTO player VALUES ('2010044','이철우','K03','LEE, CHULWOO','','2010','FW','9','','1987-11-30','1','185','78');
INSERT INTO player VALUES ('2011057','코난','K03','Goram Petreski','','2010','FW','11','','1982-05-23','1','180','76');
INSERT INTO player VALUES ('2000095','정민규','K03','JEONG, MINJYU','','2012','MF','35','','1989-09-29','1','178','71');
INSERT INTO player VALUES ('2000062','제형진','K01','JAE, HYUNGJIN','','2012','DF','38','','1989-06-25','1','179','75');
INSERT INTO player VALUES ('2000063','곽기훈','K01','KWAK, KIHOON','','2012','FW','33','','1989-11-05','1','177','72');
INSERT INTO player VALUES ('2000064','최민영','K01','CHOI, MINYOUNG','','2010','FW','37','','1991-03-07','1','174','67');
INSERT INTO player VALUES ('2000071','김회택','K07','','','','DF','37','','1991-03-07','1','174','67');
INSERT INTO player VALUES ('2000072','서현옥','K07','','','','DF','37','','1991-03-07','1','174','67');
INSERT INTO player VALUES ('2000073','정상호','K07','','','','DF','37','','1991-03-07','1','174','67');
INSERT INTO player VALUES ('2000074','최철우','K07','','','','DF','37','','1991-03-07','1','174','67');
INSERT INTO player VALUES ('2012081','정영광','K07','','','','GK','41','','1993-06-28','1','185','80');
INSERT INTO player VALUES ('2007227','최종문','K07','','','','GK','1','','1980-10-02','1','185','76');
INSERT INTO player VALUES ('2012088','염동균','K07','','','','GK','31','','1993-09-06','1','189','83');
INSERT INTO player VALUES ('2012089','김정래','K07','','','','GK','33','','1989-11-12','1','185','81');
INSERT INTO player VALUES ('2008212','강철','K07','','','','DF','3','','1981-11-02','1','178','72');
INSERT INTO player VALUES ('2012077','정강선','K07','','','','DF','37','','1989-05-23','1','180','73');
INSERT INTO player VALUES ('2012083','정인호','K07','','','','DF','39','','1993-06-09','1','184','79');
INSERT INTO player VALUES ('2007213','정태영','K07','','','','DF','7','','1980-11-08','1','180','74');
INSERT INTO player VALUES ('2007209','정현수','K07','','','','DF','21','','1983-02-14','1','176','74');
INSERT INTO player VALUES ('2012084','정형주','K07','','','','DF','21','','1983-02-14','1','176','74');
INSERT INTO player VALUES ('2012085','정홍철','K07','','','','DF','36','','1989-06-02','1','181','69');
INSERT INTO player VALUES ('2008359','마시엘','K07','','','','DF','24','','1982-03-15','1','177','70');
INSERT INTO player VALUES ('2011034','김창원','K07','','','','DF','5','','1985-07-10','1','183','75');
INSERT INTO player VALUES ('2012090','장경진','K07','','','','DF','34','','1993-08-31','1','184','82');
INSERT INTO player VALUES ('2008425','주영호','K07','','','','DF','6','','1985-10-24','1','185','80');
INSERT INTO player VALUES ('2012092','홍성요','K07','','','','DF','28','','1989-05-26','1','184','78');
INSERT INTO player VALUES ('2009115','정경일','K07','','','','MF','49','','1990-08-30','1','170','65');
INSERT INTO player VALUES ('2011035','정길식','K07','','','','MF','12','','1988-08-24','1','178','72');
INSERT INTO player VALUES ('2010030','정남일','K07','','','','MF','4','','1987-03-14','1','182','76');
INSERT INTO player VALUES ('2010032','정대욱','K07','','','','MF','18','','1988-04-02','1','182','73');
INSERT INTO player VALUES ('2007210','정도근','K07','','','','MF','10','','1982-03-02','2','181','69');
INSERT INTO player VALUES ('2012078','정동희','K07','','','','MF','38','','1993-05-06','1','174','64');
INSERT INTO player VALUES ('2007102','정명곤','K07','','','','MF','2','','1984-04-15','1','177','73');
INSERT INTO player VALUES ('2012079','정성진','K07','','','','MF','44','','1989-01-20','1','178','68');
INSERT INTO player VALUES ('2012080','정승현','K07','','','','MF','26','','1989-08-17','1','181','71');
INSERT INTO player VALUES ('2012082','정요환','K07','','','','MF','25','','1987-05-23','1','170','62');
INSERT INTO player VALUES ('2009100','정정겸','K07','','','','MF','13','','1986-06-09','2','172','65');
INSERT INTO player VALUES ('2008428','정종현','K07','','','','MF','11','','1983-07-10','1','173','68');
INSERT INTO player VALUES ('2012086','노병준','K07','','','','MF','22','','1989-09-29','1','177','67');
INSERT INTO player VALUES ('2012087','최종우','K07','','','','MF','43','','1989-04-11','1','176','69');
INSERT INTO player VALUES ('2007305','조진원','K07','','','','MF','9','','1984-09-27','1','176','75');
INSERT INTO player VALUES ('2012132','실바','K07','','','','MF','45','','1987-06-20','1','173','67');
INSERT INTO player VALUES ('2010031','윤용구','K07','','','','MF','15','','1987-08-08','1','168','60');
INSERT INTO player VALUES ('2011127','김반','K07','','','','MF','14','','1984-10-27','1','174','69');
INSERT INTO player VALUES ('2011038','김영수','K07','','','','MF','30','','1988-07-30','1','175','65');
INSERT INTO player VALUES ('2008423','임관식','K07','','','','MF','29','','1985-07-28','1','172','68');
INSERT INTO player VALUES ('2011036','이정호','K07','','','','MF','23','','1988-04-06','1','176','71');
INSERT INTO player VALUES ('2011039','하기윤','K07','','','','MF','32','','1992-03-10','1','180','65');
INSERT INTO player VALUES ('2010003','정대철','K07','','','','FW','20','','1987-08-26','1','183','78');
INSERT INTO player VALUES ('2010154','꼬레아','K07','','','','FW','16','','1984-08-23','1','172','70');
INSERT INTO player VALUES ('2007214','노상래','K07','','','','FW','8','','1980-12-15','1','175','74');
INSERT INTO player VALUES ('2009149','성한수','K07','','','','FW','40','','1986-03-10','2','177','69');
INSERT INTO player VALUES ('2009161','세자르','K07','','','','FW','17','','1983-12-09','1','180','78');
INSERT INTO player VALUES ('2012032','조병호','K07','','','','FW','27','','1987-04-26','1','177','75');
INSERT INTO player VALUES ('2011120','찌코','K07','','','','FW','27','','1985-01-26','1','180','67');
INSERT INTO player VALUES ('2010037','하성룡','K07','','','','FW','35','','1992-02-03','1','179','68');
INSERT INTO player VALUES ('2012091','홍복표','K07','','','','FW','19','','1989-10-28','1','182','73');
INSERT INTO player VALUES ('2000081','김윤환','K05','','','','DF','19','','1989-10-28','1','182','73');
INSERT INTO player VALUES ('2000082','정은철','K05','','','','DF','19','','1989-10-28','1','182','73');
INSERT INTO player VALUES ('2000083','김경춘','K05','','','','DF','19','','1989-10-28','1','182','73');
INSERT INTO player VALUES ('2000084','최동우','K05','','','','DF','19','','1989-10-28','1','182','73');
INSERT INTO player VALUES ('2000085','전명구','K05','','','','DF','19','','1989-10-28','1','182','73');
INSERT INTO player VALUES ('2000086','박경치','K05','','','','DF','19','','1989-10-28','1','182','73');
INSERT INTO player VALUES ('2007106','정이섭','K05','JEONG, ISUB','쾌남','2012','GK','45','','1984-04-06','1','185','78');
INSERT INTO player VALUES ('2012007','최관민','K05','CHOI, KWANMIN','','2012','GK','31','','1989-05-26','1','188','85');
INSERT INTO player VALUES ('2008179','최동우','K05','','','','GK','60','','1980-11-03','1','187','78');
INSERT INTO player VALUES ('2008138','김용발','K05','KIM, YONGBAL','','2004','GK','18','','1983-03-15','1','183','77');
INSERT INTO player VALUES ('2012006','김창민','K05','KIM, CHANGMIN','고릴라','2012','GK','1','','1990-01-25','1','191','87');
INSERT INTO player VALUES ('2010010','정대건','K05','JEONG, DAEKEON','','2010','DF','26','','1987-04-27','1','183','78');
INSERT INTO player VALUES ('2007159','정봉현','K05','JEONG, BONGHYUN','','2005','MF','23','','1984-07-07','1','188','74');
INSERT INTO player VALUES ('2012009','정창민','K05','JEONG, CHANGMIN','야생마','2012','DF','36','','1989-09-22','1','181','68');
INSERT INTO player VALUES ('2012005','최동혁','K05','CHOI, DONGHYUK','백호','2012','DF','23','','1989-04-18','1','185','78');
INSERT INTO player VALUES ('2012012','백인석','K05','BAEK, INSEOK','떡맨','2012','MF','40','','1989-08-08','1','179','72');
INSERT INTO player VALUES ('2008391','서혁수','K05','SEO, HYUKSU','맷돌','2008','DF','6','','1983-10-01','1','178','77');
INSERT INTO player VALUES ('2012004','안광호','K05','AN, KWANGHO','왕깡','2012','DF','15','','1989-01-10','1','190','78');
INSERT INTO player VALUES ('2009076','임종훈','K05','LIM, HONGHUN','꺼시','2009','DF','31','','1986-06-14','1','182','74');
INSERT INTO player VALUES ('2007148','이진철','K05','LEE, JINCHEUL','','2006','DF','4','','1981-03-26','1','187','75');
INSERT INTO player VALUES ('2009061','한종성','K05','HAN, JONGSUNG','','2009','DF','13','','1987-01-30','1','177','70');
INSERT INTO player VALUES ('2009160','호제리오','K05','Rogerio Prateat','저승사자','2009','DF','3','','1983-03-09','1','185','85');
INSERT INTO player VALUES ('2007031','황승주','K05','','','','DF','98','','1982-05-09','1','176','68');
INSERT INTO player VALUES ('2010023','고관영','K05','KO, KWANYOUNG','까우','2010','MF','32','','1991-12-07','1','180','72');
INSERT INTO player VALUES ('2007155','정경량','K05','JEONG, KYUNGRYANG','','2006','MF','19','','1983-12-22','1','173','65');
INSERT INTO player VALUES ('2010025','정은익','K05','','','','MF','35','','1991-03-09','1','176','63');
INSERT INTO player VALUES ('2012001','레오마르','K05','Leo03 Leiria','레오','2012','MF','5','','1981-06-26','1','183','77');
INSERT INTO player VALUES ('2008269','명재용','K05','MYUNG, JAEYOENG','','2007','MF','7','','1983-02-26','2','173','63');
INSERT INTO player VALUES ('2007149','변재섭','K05','BYUN, JAESUB','작은탱크','2007','MF','11','','1985-09-17','2','170','63');
INSERT INTO player VALUES ('2012002','보띠','K05','Raphael JoseBotti Zacarias Sena','Botti','2012','MF','10','','1991-02-23','1','174','68');
INSERT INTO player VALUES ('2011123','비에라','K05','Vieira','','2011','MF','21','','1984-02-25','1','176','73');
INSERT INTO player VALUES ('2008460','서동원','K05','SEO, DONGWON','','2008','MF','22','','1985-08-14','1','184','78');
INSERT INTO player VALUES ('2010019','안대현','K05','AN, DAEHYUN','','2010','MF','25','','1987-08-20','1','179','72');
INSERT INTO player VALUES ('2010018','양현정','K05','YANG, HYUN06G','','2010','MF','14','','1987-07-25','1','176','72');
INSERT INTO player VALUES ('2010022','유원섭','K05','YOU, WONSUOB','앙마','2010','MF','37','','1991-05-24','1','180','77');
INSERT INTO player VALUES ('2012008','김수철','K05','KIM, SUCHEUL','','2012','MF','34','','1989-05-26','1','171','68');
INSERT INTO player VALUES ('2012013','임다한','K05','LIM, DAHAN','달마','2012','DF','39','','1989-07-21','1','181','67');
INSERT INTO player VALUES ('2007087','전경준','K05','JEON, KYOUNGJOON','','2001','MF','16','','1983-09-10','1','178','68');
INSERT INTO player VALUES ('2010021','이영훈','K05','LEE, YOUNGHOON','깐돌이','2010','MF','27','','1991-03-18','1','178','71');
INSERT INTO player VALUES ('2011028','추운기','K05','CHOO, WOONKI','코알라','2011','MF','17','','1988-04-03','1','173','69');
INSERT INTO player VALUES ('2008393','하은철','K05','HA, EUNCHEUL','','2008','MF','29','','1985-06-23','1','176','69');
INSERT INTO player VALUES ('2011030','고민기','K05','GO, MINGI','','2011','FW','24','','1988-07-01','1','178','70');
INSERT INTO player VALUES ('2008143','정도훈','K05','JEONG, DOHOON','폭격기','2005','FW','9','','1980-07-21','1','182','77');
INSERT INTO player VALUES ('2012011','정연건','K05','JEONG, YEONGUN','','2012','FW','44','','1991-03-12','1','188','77');
INSERT INTO player VALUES ('2011002','정현기','K05','JEONG, HYUNKI','','2011','FW','28','','1992-03-19','1','190','82');
INSERT INTO player VALUES ('2011003','남궁도','K05','NAM, KUNGDO','','2011','FW','33','','1992-06-04','1','186','80');
INSERT INTO player VALUES ('2010101','문태혁','K05','MOON, TAEHYUK','','2012','FW','41','','1993-03-31','1','178','74');
INSERT INTO player VALUES ('2008392','최성배','K05','CHOI, SUNGBAE','흑상어','2008','FW','8','','1985-11-28','2','181','76');
INSERT INTO player VALUES ('2012125','에드밀손','K05','Edmilson','EDY','2012','FW','20','','1978-05-29','1','184','79');
INSERT INTO player VALUES ('2012003','쿠키','K05','','','','FW','99','','1981-04-30','1','170','76');
INSERT INTO player VALUES ('2000087','정은익','K05','JEONG, EUNIK','','2010','FW','35','','1991-03-09','1','176','65');
INSERT INTO player VALUES ('2000091','이순호','K03','LEE, SOONHO','','','DF','35','','1991-03-09','1','176','65');
INSERT INTO player VALUES ('2000092','유동관','K03','YOU, DONGKWAN','','','DF','35','','1991-03-09','1','176','65');
INSERT INTO player VALUES ('2000093','박기동','K03','PARK, GIDONG','','','DF','35','','1991-03-09','1','176','65');
INSERT INTO player VALUES ('2011111','김반코비','K02','','','','MF','47','','1985-02-08','1','185','78');
INSERT INTO player VALUES ('2012020','김선우','K02','KIM, SUNWOO','','','FW','33','','1988-04-01','1','174','68');
INSERT INTO player VALUES ('2011074','김여성','K02','KIM, YEOSUNG','','','MF','36','','1993-01-05','1','179','63');
INSERT INTO player VALUES ('2007064','김용우','K02','KIM, YONGWOO','','','FW','27','','1987-07-20','1','175','70');
INSERT INTO player VALUES ('2012019','김종민','K02','KIM, JONGMIN','','','MF','30','','1993-09-01','1','174','65');
INSERT INTO player VALUES ('2008437','박용훈','K02','PARK, YONGHOON','','','MF','9','','1989-03-11','1','175','67');
INSERT INTO player VALUES ('2008436','김만근','K02','KIM, MANKEUN','','','FW','34','','1987-11-28','1','177','67');
INSERT INTO player VALUES ('2007070','김재민','K02','KIM, JAEMIN','','','MF','35','','1988-05-22','1','180','71');
INSERT INTO player VALUES ('2007043','김현두','K02','KIM, HYUNDOO','','','MF','12','','1983-11-23','1','176','68');
INSERT INTO player VALUES ('2012017','이성용','K02','LEE, SUNGYONG','','','DF','20','','1985-12-25','1','173','70');
INSERT INTO player VALUES ('2012018','하태근','K02','HA, TAEKEUN','','','MF','29','','1993-07-01','1','182','75');
INSERT INTO player VALUES ('2012122','미트로','K02','Mitrovic Slavisa','','','FW','19','보스니아','1987-07-05','1','192','89');
INSERT INTO player VALUES ('2007048','최건하','K02','CHOI, KUNHA','','','FW','18','','1981-07-25','1','180','75');
INSERT INTO player VALUES ('2010168','산드로','K02','Sandro Cardoso Dos Santos','','','FW','13','브라질','1990-03-22','1','182','78');
INSERT INTO player VALUES ('2012023','윤화평','K02','YOON, HWAPYUNG','','','FW','42','','1993-03-26','1','182','74');
INSERT INTO player VALUES ('2012022','김관희','K02','KIM, KWANHEE','','','FW','39','','1993-06-29','1','180','70');
INSERT INTO player VALUES ('2000051','김광래','K09','','','','DF','39','','1993-06-29','1','180','70');
INSERT INTO player VALUES ('2000052','김춘석','K09','','','','DF','39','','1993-06-29','1','180','70');
INSERT INTO player VALUES ('2000053','조재흠','K09','','','','DF','39','','1993-06-29','1','180','70');
INSERT INTO player VALUES ('2000054','김영진','K09','','','','DF','1','','1991-05-03','1','190','89');
INSERT INTO player VALUES ('2000055','정귀화','K09','','','','DF','1','','1991-05-03','1','190','89');
INSERT INTO player VALUES ('2012094','최동석','K09','','','','GK','1','','1991-05-03','1','190','89');
INSERT INTO player VALUES ('2010125','백민철','K09','','','','GK','21','','1987-07-28','1','185','78');
INSERT INTO player VALUES ('2007236','조의손','K09','','','','GK','44','','1970-01-12','1','192','87');
INSERT INTO player VALUES ('2011018','우태식','K09','','','','GK','31','','1993-01-08','1','185','75');
INSERT INTO player VALUES ('2010117','정동진','K09','','','','DF','4','','1991-01-29','1','182','74');
INSERT INTO player VALUES ('2008414','정성일','K09','','','','DF','19','','1984-04-13','1','178','75');
INSERT INTO player VALUES ('2012093','정치곤','K09','','','','DF','17','','1993-07-29','1','183','76');
INSERT INTO player VALUES ('2012095','최요셉','K09','','','','DF','5','','1990-12-03','1','183','75');
INSERT INTO player VALUES ('2010119','최용호','K09','','','','DF','14','','1991-03-25','1','182','69');
INSERT INTO player VALUES ('2007068','최정석','K09','','','','DF','6','','1987-04-19','1','182','78');
INSERT INTO player VALUES ('2007266','손현준','K09','','','','DF','45','','1982-03-20','1','175','70');
INSERT INTO player VALUES ('2011012','안성훈','K09','','','','MF','2','','1992-09-11','1','176','63');
INSERT INTO player VALUES ('2008420','김상헌','K09','','','','DF','25','','1985-10-11','2','185','80');
INSERT INTO player VALUES ('2012102','박성호','K09','','','','DF','34','','1996-04-07','1','184','71');
INSERT INTO player VALUES ('2007225','이윤열','K09','','','','DF','3','','1984-04-17','1','185','79');
INSERT INTO player VALUES ('2009048','정도용','K09','','','','MF','40','','1986-05-28','1','168','68');
INSERT INTO player VALUES ('2011072','정민성','K09','','','','MF','24','','1992-05-28','1','175','64');
INSERT INTO player VALUES ('2011008','조재필','K09','','','','MF','41','','1992-05-25','1','180','70');
INSERT INTO player VALUES ('2012096','조현국','K09','','','','MF','37','','1993-03-21','1','172','65');
INSERT INTO player VALUES ('2010049','안드레','K09','','','','MF','18','브라질','1982-11-16','1','172','69');
INSERT INTO player VALUES ('2012097','안상현','K09','','','','MF','33','','1996-03-05','1','176','67');
INSERT INTO player VALUES ('2012098','우진석','K09','','','','MF','39','','1989-08-26','1','176','70');
INSERT INTO player VALUES ('2007133','유상수','K09','','','','MF','43','','1983-08-28','2','175','67');
INSERT INTO player VALUES ('2010048','김영표','K09','','','','MF','12','','1987-04-23','1','176','66');
INSERT INTO player VALUES ('2012100','김준기','K09','','','','MF','36','','1992-04-25','1','183','74');
INSERT INTO player VALUES ('2012101','김지남','K09','','','','MF','42','','1994-11-21','1','183','68');
INSERT INTO player VALUES ('2008413','박광민','K09','','','','FW','23','','1986-01-08','1','180','70');
INSERT INTO player VALUES ('2009051','박창근','K09','','','','MF','27','','1993-08-10','2','170','63');
INSERT INTO player VALUES ('2010118','이원권','K09','','','','MF','13','','1991-11-08','1','173','65');
INSERT INTO player VALUES ('2010120','정병채','K09','','','','FW','15','','1991-04-14','1','180','72');
INSERT INTO player VALUES ('2009046','정성재','K09','','','','MF','8','','1986-09-17','2','178','72');
INSERT INTO player VALUES ('2012119','뚜따','K09','','','','FW','99','브라질','1984-06-20','1','187','84');
INSERT INTO player VALUES ('2012105','마르코','K09','','','','FW','11','브라질','1986-12-03','1','175','77');
INSERT INTO player VALUES ('2011005','최성호','K09','','','','FW','30','','1992-07-27','1','187','73');
INSERT INTO player VALUES ('2011108','최윤화','K09','','','','MF','16','','1988-06-13','1','175','67');
INSERT INTO player VALUES ('2009047','최정환','K09','','','','FW','20','','1987-01-14','1','178','70');
INSERT INTO player VALUES ('2009005','왕정현','K09','','','','FW','28','','1986-08-30','2','181','73');
INSERT INTO player VALUES ('2012099','김정수','K09','','','','FW','7','','1990-01-08','1','184','74');
INSERT INTO player VALUES ('2012103','김원광','K09','','','','FW','22','','1995-08-23','1','183','68');
INSERT INTO player VALUES ('2009044','진순진','K09','','','','MF','26','','1984-03-01','2','180','78');
INSERT INTO player VALUES ('2010053','이태욱','K09','','','','FW','9','','1991-03-13','1','173','66');
INSERT INTO player VALUES ('2012104','한동원','K09','','','','FW','32','','1996-04-06','1','174','61');
INSERT INTO player VALUES ('2011004','한정화','K09','','','','FW','29','','1992-10-31','1','173','63');
INSERT INTO player VALUES ('2010164','히카르도','K09','','','','MF','10','브라질','1986-06-08','1','189','71');
INSERT INTO player VALUES ('2011082','권정혁','K01','KWON, 06GHYUK','','2011','GK','1','','1988-08-02','1','195','80');
INSERT INTO player VALUES ('2009070','최창주','K01','CHOI, CHANGZOO','','2009','GK','40','','1982-09-30','1','187','86');
INSERT INTO player VALUES ('2007020','서동명','K01','SEO, DONGMYUNG','','2012','GK','21','','1984-05-04','1','196','94');
INSERT INTO player VALUES ('2008410','양지원','K01','YANG, JIWON','','2008','GK','45','','1984-04-28','1','181','75');
INSERT INTO player VALUES ('2012039','이무림','K01','LEE, MOOLIM','','2012','GK','31','','1989-04-15','1','185','79');
INSERT INTO player VALUES ('2012034','정윤구','K01','JEONG, YOONGOO','','2012','DF','20','','1989-09-01','1','182','75');
INSERT INTO player VALUES ('2011106','끌레베르','K01','Cleber Arild','','2011','DF','3','브라질','1979-01-21','1','186','79');
INSERT INTO player VALUES ('2011080','최경삼','K01','CHOI, GYUNGSAM','','2011','DF','25','','1988-06-06','1','188','74');
INSERT INTO player VALUES ('2007004','최종욱','K01','CHOI, JONGWOOK','','2007','DF','29','','1985-01-11','1','185','74');
INSERT INTO player VALUES ('2012035','변성환','K01','BYUN, SUNGHWAN','','2012','DF','23','','1989-12-22','1','176','69');
INSERT INTO player VALUES ('2008366','손상호','K01','SON, SANGHO','','2007','DF','28','','1984-05-04','1','187','78');
INSERT INTO player VALUES ('2012044','유재형','K01','YOU, JAEHYUNG','','2012','DF','39','','1987-08-24','1','180','79');
INSERT INTO player VALUES ('2011077','김세권','K01','KIM, SAEKWON','','2011','DF','4','','1988-06-26','1','184','80');
INSERT INTO player VALUES ('2009069','하성용','K01','HA, SUNGYONG','','2009','DF','43','','1986-10-05','1','179','75');
INSERT INTO player VALUES ('2010082','한정일','K01','HAN, JEUNGIL','','2010','DF','41','','1987-03-03','1','172','65');
INSERT INTO player VALUES ('2010054','정건형','K01','JEONG, KUNHYUNG','','2010','MF','17','','1989-09-11','1','176','70');
INSERT INTO player VALUES ('2009058','정도균','K01','JEONG, DOKYUN','','2009','MF','6','','1987-01-13','1','184','78');
INSERT INTO player VALUES ('2011083','최기욱','K01','CHOI, KIWOOK','','2011','MF','18','','1988-12-22','1','175','68');
INSERT INTO player VALUES ('2012028','최진섭','K01','CHOI, JINSUP','','2012','MF','2','','1987-03-11','1','179','68');
INSERT INTO player VALUES ('2011079','서덕규','K01','SEO, DOUKGUY','','2011','DF','5','','1988-10-22','1','182','72');
INSERT INTO player VALUES ('2012130','에디','K01','Edmilson Alves','','2012','MF','7','브라질','1986-02-17','1','182','70');
INSERT INTO player VALUES ('2009055','김길용','K01','KIM, KILYONG','','2009','MF','22','','1986-03-30','2','183','78');
INSERT INTO player VALUES ('2012036','김윤섭','K01','KIM, YOONSUB','','2012','MF','26','','1989-07-30','1','184','77');
INSERT INTO player VALUES ('2012029','김천수','K01','KIM, CHUNSU','','2012','MF','9','','1991-07-09','1','174','65');
INSERT INTO player VALUES ('2012033','장성욱','K01','01G, SUNGWOOK','','2012','MF','19','','1989-09-01','1','174','71');
INSERT INTO player VALUES ('2007023','장철민','K01','01G, CHULMIN','','2007','MF','24','','1982-05-19','1','179','76');
INSERT INTO player VALUES ('2011084','전광철','K01','JEON, KWANGCHEOL','','2011','MF','34','','1992-07-16','1','186','74');
INSERT INTO player VALUES ('2012045','전재운','K01','JEON, JAEWOON','','2012','MF','42','','1991-06-05','1','177','67');
INSERT INTO player VALUES ('2007008','박정수','K01','PARK, JEONGSOO','','2003','MF','8','','1979-11-20','1','170','61');
INSERT INTO player VALUES ('2012042','박주영','K01','PARK, HUYOUNG','','2012','MF','36','','1989-05-03','1','172','65');
INSERT INTO player VALUES ('2012040','김원희','K01','KIM, WONHEE','','2012','MF','32','','1993-04-17','1','178','71');
INSERT INTO player VALUES ('2012041','곽기훈','K01','','','','FW','33','','1989-11-05','1','177','72');
INSERT INTO player VALUES ('2007010','정현석','K01','JEONG, HYUNSEOK','','2011','FW','16','','1977-05-05','1','178','70');
INSERT INTO player VALUES ('2011105','마르코스','K01','03cos Antonio','','2011','FW','44','브라질','1987-04-07','1','170','73');
INSERT INTO player VALUES ('2010045','최규선','K01','CHOI, GYUSUN','','2010','FW','15','','1991-09-24','1','182','66');
INSERT INTO player VALUES ('2012129','알리송','K01','Alison Barros Moraes','','','FW','14','브라질','1992-06-30','1','171','60');
INSERT INTO player VALUES ('2012038','김태유','K01','KIM, TAEYOO','','2012','FW','30','','1989-10-01','1','175','64');
INSERT INTO player VALUES ('2012030','박성훈','K01','','','','FW','10','','1989-07-04','1','187','83');
INSERT INTO player VALUES ('2012037','이인석','K01','LEE, INSUK','','2012','FW','27','','1989-08-07','1','181','74');
INSERT INTO player VALUES ('2011104','파울링뇨','K01','03cos Paulo Paulrini','','2011','FW','11','브라질','1987-03-04','1','174','75');
INSERT INTO player VALUES ('2012031','현영민','K01','HYUN, YOUNGMIN','','2012','FW','13','','1989-12-25','1','179','73');
INSERT INTO player VALUES ('2000061','홍인기','K01','HONG, INKI','','2008','DF','35','','1989-03-15','1','182','75');
INSERT INTO player VALUES ('2007114','정기동','K04','','','','MF','6','','1981-01-12','1','171','66');
INSERT INTO player VALUES ('2010005','정기형','K04','','','','MF','19','','1987-07-10','1','173','64');
INSERT INTO player VALUES ('2012055','정정현','K04','','','','MF','32','','1989-04-01','1','176','68');
INSERT INTO player VALUES ('2007122','정한윤','K04','','','','MF','14','','1984-07-11','1','185','74');
INSERT INTO player VALUES ('2007137','남기일','K04','','','','MF','29','','1984-08-17','1','177','69');
INSERT INTO player VALUES ('2012131','무스타파','K04','','','','MF','77','','1985-01-08','1','180','73');
INSERT INTO player VALUES ('2012056','최정주','K04','','','','MF','33','','1989-06-26','1','177','73');
INSERT INTO player VALUES ('2010004','조현호','K04','','','','MF','22','','1987-07-07','1','183','71');
INSERT INTO player VALUES ('2007125','윤정춘','K04','','','','MF','17','','1983-02-18','1','173','66');
INSERT INTO player VALUES ('2011103','장경호','K04','','','','MF','39','','1993-02-23','1','174','66');
INSERT INTO player VALUES ('2008209','이문식','K04','','','','MF','10','','1981-10-23','1','173','66');
INSERT INTO player VALUES ('2008457','곽경근','K04','GWAK, JYUNGKEUN','골든맨','2008','FW','9','','1982-10-10','1','184','81');
INSERT INTO player VALUES ('2012120','다보','K04','Chelck Ou03 Dabo','','2012','FW','9','','1982-10-10','1','184','81');
INSERT INTO player VALUES ('2007126','최성철','K04','CHOI, SUNGCHUL','코미디언','2007','FW','28','','1985-03-16','1','190','78');
INSERT INTO player VALUES ('2007153','비탈리','K04','VITALIY','','2012','FW','24','','1979-05-04','1','183','74');
INSERT INTO player VALUES ('2009026','안승인','K04','AN, SEUNGIN','안뽕','2009','FW','13','','1983-03-14','1','182','75');
INSERT INTO player VALUES ('2011019','윤원철','K04','','','','FW','8','','1989-01-06','1','184','75');
INSERT INTO player VALUES ('2009027','김성재','K04','KIM, SUNGJAE','까치,돌쇠','2009','FW','11','','1986-05-16','1','178','67');
INSERT INTO player VALUES ('2007135','김원식','K04','KIM, WONSHIK','뺀, 밥식이','2006','FW','27','','1983-05-16','1','172','64');
INSERT INTO player VALUES ('2012057','김종민','K04','KIM, JONGMIN','','2012','FW','34','','1993-08-01','1','184','77');
INSERT INTO player VALUES ('2011020','박필석','K04','PARK, PILSUK','장신','2011','FW','18','','1988-07-23','1','190','83');
INSERT INTO player VALUES ('2010084','제프유','K04','JEFF JI YOUNG','','2011','FW','16','미국','1988-10-30','1','184','76');
INSERT INTO player VALUES ('2000026','롤란','K04','','','2010','FW','16','미국','1988-10-30','1','184','76');
INSERT INTO player VALUES ('2000027','셀라하틴','K04','Selahattin','','2012','FW','16','미국','1988-10-30','1','184','76');
INSERT INTO player VALUES ('2000031','차경복','K08','Selahattin','','2012','FW','16','미국','1988-10-30','1','184','76');
INSERT INTO player VALUES ('2000032','정학범','K08','Selahattin','','2012','FW','16','미국','1988-10-30','1','184','76');
INSERT INTO player VALUES ('2000033','안익수','K08','Selahattin','','2012','FW','16','미국','1988-10-30','1','184','76');
INSERT INTO player VALUES ('2000034','차상광','K08','Selahattin','','2012','FW','16','미국','1988-10-30','1','184','76');
INSERT INTO player VALUES ('2009045','권찬수','K08','KWON, CHANSOO','','','GK','21','','1984-05-30','2','183','77');
INSERT INTO player VALUES ('2012107','정경두','K08','KYOUNGDOO','','','GK','21','','1984-05-30','2','183','77');
INSERT INTO player VALUES ('2007255','정해운','K08','JEONG, HAEWOON','','','GK','1','','1983-12-25','2','185','79');
INSERT INTO player VALUES ('2009101','양영민','K08','YANG, YOUNGMIN','','','GK','31','','1984-07-19','1','190','80');
INSERT INTO player VALUES ('2011087','가이모토','K08','KAIMOTO KOJIRO','','','DF','3','','1987-10-14','1','182','77');
INSERT INTO player VALUES ('2012108','정두영','K08','JEONG, DOYOUNG','','','DF','30','','1990-10-07','1','181','75');
INSERT INTO player VALUES ('2012118','정명휘','K08','JEONG, MYONGHWI','','','DF','37','','1991-05-08','1','185','76');
INSERT INTO player VALUES ('2007329','정영철','K08','JEONG, YOUNGCHUL','','','DF','5','','1986-06-30','1','183','80');
INSERT INTO player VALUES ('2007259','정정재','K08','JEONG, 06GJAE','','','DF','13','','1984-05-22','2','184','80');
INSERT INTO player VALUES ('2007186','정현수','K08','JEONG, HYUNSOO','','','DF','15','','1983-03-13','2','186','80');
INSERT INTO player VALUES ('2007089','노태경','K08','NO, TAEKYOUNG','','','DF','25','','1982-04-22','1','175','68');
INSERT INTO player VALUES ('2009098','문삼진','K08','MOON, SAMJIN','','','DF','2','','1983-03-03','1','173','68');
INSERT INTO player VALUES ('2012111','민경인','K08','MIN, KYUNGIN','','','DF','12','','1989-05-09','1','177','71');
INSERT INTO player VALUES ('2011091','최종남','K08','CHOI, JONGNAM','','','DF','34','','1986-11-17','1','182','76');
INSERT INTO player VALUES ('2007055','최충균','K08','CHOI, CHOONGKYUN','','','DF','20','','1983-06-20','1','184','70');
INSERT INTO player VALUES ('2010034','배진수','K08','BAE, JINSOO','','','DF','38','','1986-01-25','1','178','72');
INSERT INTO player VALUES ('2008471','김영진','K08','KIM, YOUNGJIN','','','DF','23','','1982-03-27','2','180','75');
INSERT INTO player VALUES ('2008399','임규식','K08','LIM, KYUSIK','','','DF','27','','1985-05-09','1','182','75');
INSERT INTO player VALUES ('2012106','곽치국','K08','KWAK, CHIGOOG','','','MF','32','','1989-04-22','1','170','68');
INSERT INTO player VALUES ('2009095','정상식','K08','JEONG, SANGSIK','','','MF','4','','1986-12-17','1','184','71');
INSERT INTO player VALUES ('2011086','정용희','K08','JEONG, YONGHEE','','','MF','18','','1988-10-15','1','178','70');
INSERT INTO player VALUES ('2012109','노규영','K08','','','','MF','33','','1988-04-29','1','180','83');
INSERT INTO player VALUES ('2010126','최강조','K08','CHOI, KANGJO','','','MF','6','','1990-01-24','1','165','57');
INSERT INTO player VALUES ('2008171','최남열','K08','CHOI, NAMRYUL','','','MF','8','','1980-05-07','2','178','70');
INSERT INTO player VALUES ('2011088','백영철','K08','BAEK, YOUNGCHUL','','','MF','22','','1988-11-11','1','177','71');
INSERT INTO player VALUES ('2007242','조태용','K08','CHO, TAEYONG','','','MF','7','','1980-10-11','1','174','67');
INSERT INTO player VALUES ('2012117','올리베','K08','Alcir de Oliveira fonseca','','','MF','29','브라질','1987-11-14','1','175','73');
INSERT INTO player VALUES ('2011128','김리네','K08','Ricardo Irineu','','','MF','26','브라질','1987-07-12','1','172','69');
INSERT INTO player VALUES ('2012128','쟈스민','K08','Jasmin Mujidza','','','MF','33','크로아티아','1984-03-02','1','180','78');
INSERT INTO player VALUES ('2012114','전재호','K08','CHUN, JAEHO','','','MF','14','','1989-08-08','1','168','64');
INSERT INTO player VALUES ('2008131','김진호','K08','KIM, JINHO','','','MF','11','','1983-08-02','1','174','68');
INSERT INTO player VALUES ('2012115','이낙훈','K08','LEE, NAKHOON','','','MF','35','','1989-12-28','1','172','62');
INSERT INTO player VALUES ('2007100','홍도표','K08','HONG, DOPYO','','','MF','9','','1983-07-24','1','173','66');
INSERT INTO player VALUES ('2010127','정대의','K08','JEONG, DAEEUI','','','FW','19','','1987-05-30','1','174','70');
INSERT INTO player VALUES ('2009084','정우재','K08','JEONG, WOOJAE','','','FW','24','','1986-09-13','1','173','71');
INSERT INTO player VALUES ('2007192','샤샤','K08','Drakulic sasa','','','FW','10','','1982-08-28','1','190','84');
INSERT INTO player VALUES ('2012112','서관수','K08','SEO, KWANSOO','','','FW','17','','1990-02-25','1','179','74');
INSERT INTO player VALUES ('2011095','김성운','K08','KIM, SUNGWOON','','','FW','40','','1988-12-25','1','173','64');
INSERT INTO player VALUES ('2012113','김정운','K08','KIM, 06GWOON','','','FW','28','','1993-03-14','1','170','68');
INSERT INTO player VALUES ('2011094','장동현','K08','01G, DONGHYUN','','','FW','39','','1992-03-19','1','178','67');
INSERT INTO player VALUES ('2007250','황연석','K08','HWANG, YOUNSUCK','','','FW','16','','1983-10-17','1','192','81');
INSERT INTO player VALUES ('2000041','정호','K02','JEONG, HO','','','FW','16','','1983-10-17','1','192','81');
INSERT INTO player VALUES ('2000042','왕선재','K02','WANG, SUNJAE','','','FW','16','','1983-10-17','1','192','81');
INSERT INTO player VALUES ('2000043','코샤','K02','03co Antonio Alvarez Ferreira','','','FW','16','','1983-10-17','1','192','81');
INSERT INTO player VALUES ('2000044','윤성효','K02','YOON, SUNGHYO','','','FW','16','','1983-10-17','1','192','81');
INSERT INTO player VALUES ('2007067','정광수','K02','JEONG, GWANGSOO','','','GK','41','','1987-03-10','1','182','79');
INSERT INTO player VALUES ('2009004','최호진','K02','CHOI, HOJIN','','','GK','31','','1986-10-22','1','190','82');
INSERT INTO player VALUES ('2007203','조범철','K02','CHO, BUMCHUL','','','GK','21','','1980-09-27','2','185','85');
INSERT INTO player VALUES ('2007035','김운재','K02','KIM, WOONJAE','','','GK','1','','1983-04-26','1','182','82');
INSERT INTO player VALUES ('2008429','정영선','K02','JEONG, YOUNGSUN','','','DF','16','','1985-04-03','1','184','81');
INSERT INTO player VALUES ('2012021','정유진','K02','JEONG, YOUJIN','','','DF','37','','1993-06-19','1','188','76');
INSERT INTO player VALUES ('2011073','손승준','K02','SON, SEUNG06','','','DF','32','','1992-05-16','1','186','72');
INSERT INTO player VALUES ('2007007','조홍기','K02','CHO, HONGGI','','','DF','15','','1978-05-04','1','175','66');
INSERT INTO player VALUES ('2012027','안동원','K02','AN, DONGWON','','','DF','49','','1992-09-27','1','184','78');
INSERT INTO player VALUES ('2012024','김강진','K02','KIM, KANGJIN','','','DF','43','','1996-04-25','1','181','65');
INSERT INTO player VALUES ('2007036','김병근','K02','KIM, BYOUNGKEUN','','','DF','3','','1983-04-28','1','175','63');
INSERT INTO player VALUES ('2012026','김태권','K02','KIM, TAE-KWON','','','DF','46','','1990-07-14','1','180','75');
INSERT INTO player VALUES ('2012014','김병국','K02','KIM, BYUNGKOOK','','','DF','2','','1991-07-01','1','183','78');
INSERT INTO player VALUES ('2011075','김성환','K02','KIM, SUNGHWAN','','','DF','5','','1992-04-09','1','183','75');
INSERT INTO player VALUES ('2012123','가비','K02','Gabriel Popescu','','','MF','10','루마니아','1983-12-25','1','177','74');
INSERT INTO player VALUES ('2010089','강대희','K02','','','','MF','26','','1987-02-02','1','174','65');
INSERT INTO player VALUES ('2007051','고종수','K02','KO, JONGSU','','','MF','22','','1988-10-30','1','176','70');
INSERT INTO player VALUES ('2012015','고창현','K02','KO, CHANGHYUN','','','MF','8','','1993-09-15','1','170','64');
INSERT INTO player VALUES ('2009089','정기범','K02','JEONG, KIBUM','','','MF','28','','1986-08-14','1','173','65');
INSERT INTO player VALUES ('2009083','정동현','K02','JEONG, DONGHYUN','','','MF','25','','1990-08-17','1','175','72');
INSERT INTO player VALUES ('2011071','정두현','K02','JEONG, DOOHYUN','','','MF','4','','1992-07-14','1','175','67');
INSERT INTO player VALUES ('2012025','정준','K02','JEONG, 06','','','MF','44','','1996-12-09','1','170','65');
INSERT INTO player VALUES ('2007040','정진우','K02','JEONG, JINWOO','','','DF','7','','1985-10-09','1','179','72');
INSERT INTO player VALUES ('2007069','데니스','K02','Laktio11 Denis','','','FW','11','러시아','1987-09-04','1','176','71');
INSERT INTO player VALUES ('2007274','서정원','K02','SEO, 06GWON','','','FW','14','','1980-12-17','1','173','66');
INSERT INTO player VALUES ('2012016','손대호','K02','SON, DAEHO','','','DF','17','','1991-09-11','1','186','79');
INSERT INTO player VALUES ('2011070','오규찬','K02','OH, GUCHAN','','','MF','24','','1992-08-28','1','178','71');
INSERT INTO player VALUES ('2012135','윤원일','K02','','','','MF','45','','1993-03-31','1','176','70');
INSERT INTO player VALUES ('2009131','김동욱','K02','KIM, DONGWOOK','','','MF','40','','1986-04-10','1','176','68');
INSERT INTO player VALUES ('2010901','장성철','K11','','','','MF','27','','1991-03-09','1','176','75');
INSERT INTO player VALUES ('2010902','장윤정','K11','','','','DF','17','','1993-03-14','1','173','63');
INSERT INTO player VALUES ('2010903','장서연','K11','','','','FW','07','','1995-01-29','1','180','60');
INSERT INTO player VALUES ('2009901','고규억','K12','','','','DF','29','','1994-08-14','1','175','70');
INSERT INTO player VALUES ('2009902','장재우','K12','','','','FW','12','','1987-02-28','1','172','63');
INSERT INTO player VALUES ('2008901','오재진','K13','','','','DF','49','','1992-07-23','1','173','72');
INSERT INTO player VALUES ('2008902','선원길','K13','','','','GK','46','','1996-04-26','1','174','66');
INSERT INTO player VALUES ('2008903','안소정','K13','','','','FW','38','','1997-10-24','1','174','63');
INSERT INTO player VALUES ('2007901','주시형','K14','','','','DF','33','','1992-11-11','1','170','72');
INSERT INTO player VALUES ('2007902','홍종하','K14','','','','MF','32','','1988-12-21','1','169','74');
INSERT INTO player VALUES ('2010801','오춘식','K15','','','','MF','22','','1988-03-08','1','168','75');
INSERT INTO player VALUES ('2010802','박창우','K15','','','','DF','13','','1989-05-17','1','177','68');
INSERT INTO player VALUES ('2010803','박진하','K15','','','','FW','11','','1989-01-09','1','181','62');
commit;