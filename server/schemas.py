from typing import List, Optional
from datetime import datetime, date

from pydantic import BaseModel

class USER(BaseModel):
    userid: int
    name: str
    age: int
    gender: str

class USER_TESTDAY(BaseModel):
    userid: int
    day: date

class QUESTION(BaseModel):
    type: str
    sentence: str
    keyword: int

class TALK_REPORT(BaseModel):
    userid: int
    day: date
    keyword: str
    emo: int=0
    text: str

class REPORT_DIC(BaseModel):
    keyword: int
    emotion: int
    sentence: str

class DRAW_REPORT(BaseModel):
    userid: int
    day: date
    house_img: str
    tree_img: str
    person_img: str
    result: str
    hosue_text: str
    tree_text: str
    person_text: str
    f_type1: float
    f_type2: float
    f_type3: float

class CHATTING(BaseModel):
    userid: int
    day: date
    time: datetime
    type: str
    chat: str

class DRAW_RESULT(BaseModel):
    userid: int
    sentence: str

class KEYWORD(BaseModel):
    userid: int
    day: date
    keyword_index: str