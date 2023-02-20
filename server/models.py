from sqlalchemy import Column, ForeignKey, Integer, Text, VARCHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, date

Base = declarative_base()


class USER(Base):
    __tablename__ = "user"

    userid: Column(Integer, Primary_key=True, autoincrement=True, nullable=False)
    name: Column(VARCHAR(10), nullable=False)
    age: Column(Integer, nullable=False)
    gender: Column(VARCHAR(15), nullable=False)

    User_Testday = relationship("USER_TESTDAY", primaryjoin="USER.userid == USER_TESTDAY.userid", backref="user")
    Talk_Report = relationship("TALK_REPORT", primaryjoin="USER.userid == TALK_REPORT.userid", backref="user")
    Draw_Report = relationship("DRAW_REPORT", primaryjoin="USER.userid == DRAW_REPORT.userid", backref="user")
    Chatting = relationship("CHATTING", primaryjoin="USER.userid == CHATTING.userid", backref="user")

class USER_TESTDAY(Base):
    __tablename__ = "user_testday"

    userid: Column(Integer, ForeignKey("user.userid", ondelete="CASCADE", onupdate="CASCADE"))
    day: Column(date, ForeignKey("draw_report.day", ondelete="CASCADE", onupdate="CASCADE"))

    Talk_Report = relationship("TALK_REPORT", primaryjoin="USER_TESTDAY.day == TALK_REPORT.day", backref="user_testday")
    Chatting = relationship("CHATTING", primaryjoin="USER_TESTDAY.day == CHATTING.day", backref="user_testday")

class QUESTION(Base):
    __tablename__ = "question"

    type: Column(VARCHAR(7))
    sentence: Column(VARCHAR(100))
    keyword: Column(Integer)

class TALK_REPORT(Base):
    __tablename__ = "talk_report"

    userid: Column(Integer, ForeignKey("user.userid", ondelete="CASCADE", onupdate="CASCADE"))
    day: Column(date, ForeignKey("user_testday.day", ondelete="CASCADE", onupdate="CASCADE"))
    keyword: Column(VARCHAR(100))
    emo: Column(Integer = 0)
    text: Column(VARCHAR(200))

class REPORT_DIC(Base):
    __tablename__ = "report_dic" 

    keyword: Column(Integer)
    emotion: Column(Integer)
    sentence: Column(Text)

class DRAW_REPORT(Base):
    __tablename__ = "draw_report"

    userid: Column(Integer, ForeignKey("user.userid", ondelete="CASCADE", onupdate="CASCADE"))
    day: Column(datetime.date, Primary_key=True, nullable=False)
    house_img: Column(VARCHAR(50))
    tree_img: Column(VARCHAR(50))
    person_img: Column(VARCHAR(50))
    result: Column(Text)
    hosue_text: Column(VARCHAR(200))
    tree_text: Column(VARCHAR(200))
    person_text: Column(VARCHAR(200))
    f_type1: Column(Integer)
    f_type2: Column(Integer)
    f_type3: Column(Integer)

    User_Testday = relationship("USER_TESTDAY", primaryjoin="DRAW_REPORT.day == USER_TESTDAY.day", backref="draw_report")

class CHATTING(Base):
    __tablename__ = "chatting"

    userid: Column(Integer, ForeignKey("user.userid", ondelete="CASCADE", onupdate="CASCADE"))
    day: Column(date, ForeignKey("user_testday.day", ondelete="CASCADE", onupdate="CASCADE"))
    time: Column(datetime.time)
    type: Column(VARCHAR(4))
    chat: Column(VARCHAR(100))
    
class DRAW_RESULT(Base):
    __tablename__ = "draw_result"

    idx: Column(Integer, Primary_key=True)
    sentence: Column(Text)