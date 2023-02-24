
import house_output, tree_output, person_output
import func, fuzzy

def draw_result_index(h_output, t_output, p_output):
    sample_index = [[6, 7],[14, 15],[16, 17, 18],[19, 20, 21],[1, 2, 3],[4, 5],[11, 12, 13],[8, 9, 10],[32, 33, 34],[22, 23],[30, 31],[35, 36, 37],[28, 29],[24, 25],[26, 27],[38, 39, 40],[41, 42],[43, 44],[45, 46, 47],[48, 49, 50, 51],[52, 53, 54],[55, 56],[57, 58],[59, 60, 61],[62, 63, 64]]

    zero_list = [0, 0.2, 'roof1', 'Small', 'top', 'up']
    one_list = [1, 'roof2', 'Midium', 'down', 'mouth_open']
    two_list = [2, 'roof3', 'Big', 3, 'bottom', 'mouth_close']

    ht_output = dict(h_output, **t_output)
    htp_output = dict(ht_output, **p_output)
    del htp_output['roof_absence']

    output_index = []
    for idx, value in enumerate(htp_output.values()):
        if value in zero_list:
            output_index.append(sample_index[idx][0])
        elif value in one_list:
            output_index.append(sample_index[idx][1])
        elif value in two_list:
            output_index.append(sample_index[idx][2])
        elif value == 'mouth_teeth':
            output_index.append(sample_index[idx][3])
    
    return output_index

def get_keyword(output):
    keyword_list = list(set(output))
    keyword_result = []
    for i in keyword_list:
        if output.count(i) > 1:
            keyword_result.append(i)

    if not len(keyword_result):
        keyword_result = keyword_list

    return keyword_result


'''
# type1 = [house_size_value, head_size_value, nose_size_value, window_size_value, chimney_value, nose_value]
# type2 = [root_exist_value, slub_exist_value, legs_exist_value]
# type3 = [tree_size_value, trunk_size_value, nose_size_value]

h_output, h_keyword, h_sentence, house_size_value, window_size_value, chimney_value = house_output.house_print('test_images/house/reshape_test3.jpg')
t_output, t_keyword, t_sentence, root_exist_value, slub_exist_value, tree_size_value, trunk_size_value = tree_output.tree_print('test_images/tree/reshape_tree1.jpg')
p_output, p_keyword, p_sentence, head_size_value, nose_value, legs_exist_value, nose_size_value = person_output.person_print('test_images\person\KakaoTalk_20230218_140105547.jpg')

#--------fuzzy output--------#
f_type1 = fuzzy.fuzzy_type1()
type1_result = f_type1.make_decision(house_size_value, head_size_value, nose_size_value, window_size_value, chimney_value, nose_value)
f_type2 = fuzzy.fuzzy_type2()
type2_result = f_type2.make_decision(root_exist_value, slub_exist_value, legs_exist_value)
f_type3 = fuzzy.fuzzy_type3()
type3_result = f_type3.make_decision(tree_size_value, trunk_size_value, nose_size_value)

print('type1: %.2f%%, type2: %.2f%%, type3: %.2f%%' % (type1_result, type2_result, type3_result))
print('draw report index: ', draw_result_index(h_output, t_output, p_output))
print('house_keyword: ', h_keyword, 'tree_keyword: ', t_keyword, 'person_keyword: ', p_keyword, sep='\n')
'''

#---------------------server----------------------#
import os, io, json, pymysql, random
from fastapi import FastAPI, HTTPException, File, UploadFile
from datetime import datetime, date
# from server.models import *
from typing import List
from uuid import UUID 
from PIL import Image

app = FastAPI()
conn = pymysql.connect(host='jaminidb2.mysql.database.azure.com', user='sktflyaiambition4', password='rmaWhrdl8!', db='jamini', charset='utf8', ssl={"fake_flag_to_enable_tls":True})
db = conn.cursor()

#프론트에서 이미지 받기
@app.post("/uploadfilehouse")
async def create_upload_file(image: UploadFile = File(...)):
    contents = await image.read()
    with Image.open(io.BytesIO(contents)) as img:
        img = img.convert("RGB") 
        day = datetime.now().date()
        save_path = os.path.join(os.path.expanduser("~"), "server", "test_images", "house", "image_house_" + str(day) + ".jpg")  
        img.save(save_path, "JPEG")  
    return {"filename" : image.filename}

@app.post("/uploadfiletree")
async def create_upload_file(image: UploadFile = File(...)):
    contents = await image.read()
    with Image.open(io.BytesIO(contents)) as img:
        img = img.convert("RGB") 
        day = datetime.now().date()
        save_path = os.path.join(os.path.expanduser("~"), "server", "test_images", "tree", "image_tree_" + str(day) + ".jpg")  
        img.save(save_path, "JPEG")  
    return {"filename" : image.filename}

@app.post("/uploadfileperson")
async def create_upload_file(image: UploadFile = File(...)):
    contents = await image.read()
    with Image.open(io.BytesIO(contents)) as img:
        img = img.convert("RGB")  
        day = datetime.now().date()
        save_path = os.path.join(os.path.expanduser("~"), "server", "test_images", "person", "image_person_" + str(day) + ".jpg") 
        img.save(save_path, "JPEG")  
    return {"filename" : image.filename}


@app.get("/model/{user_id}")
async def main_test(user_id:int):
    conn = pymysql.connect(host='jaminidb2.mysql.database.azure.com', user='sktflyaiambition4', password='rmaWhrdl8!', db='jamini', charset='utf8', ssl={"fake_flag_to_enable_tls":True})
    db = conn.cursor()
    try:
        userid = user_id
        # sql_id = f"SELECT userid FROM USER;"
        # db.execute(sql_id)
        # db_id = db.fetchall()
            
        # # db:USER에 저장된 userid인지 확인
        # user_list = []
        # for i in range(len(db_id)):
        #     user_list.append(db_id[i][0])

        if userid != 1:
           return "아이디가 틀렸습니다."

        # 테스트 한 날짜를 db:USER_TESTDAY에 저장
        day = datetime.now().date()

        # user가 오늘 test한지 확인
        sql_day = f"SELECT day FROM USER_TESTDAY WHERE userid = {userid};"
        db.execute(sql_day)
        first_day = db.fetchall()

        day_list = []
        for i in range(len(first_day)):
            day_list.append(str(first_day[i][0]))
        
        print(day_list)

        if str(day) in day_list:
            sql_date = f"UPDATE USER_TESTDAY SET day = '{day}' WHERE userid = {userid};"
            db.execute(sql_date)
            conn.commit()
            # conn.close()

            # img를 서버에 /test_image 폴더에 저장하기
            h_img = 'test_images/house/image_house_' + str(day) + '.jpg'
            t_img = 'test_images/tree/image_tree_' + str(day) + '.jpg'
            p_img = 'test_images/person/image_person_' + str(day) + '.jpg'
            # h_img = 'test_images/house/reshape_test7.jpg'
            # t_img = 'test_images/person/test3.JPG'
            # p_img = 'test_images/tree/reshape_tree4.jpg'
            
            # model 실행
            h_output, h_keyword, h_sentence, house_size_value, window_size_value, chimney_value = house_output.house_print(h_img)
            t_output, t_keyword, t_sentence, root_exist_value, slub_exist_value, tree_size_value, trunk_size_value = tree_output.tree_print(t_img)
            p_output, p_keyword, p_sentence, head_size_value, nose_value, legs_exist_value, nose_size_value = person_output.person_print(p_img)

            f_type1 = fuzzy.fuzzy_type1()
            type1_result = f_type1.make_decision(house_size_value, head_size_value, nose_size_value, window_size_value, chimney_value, nose_value)
            f_type2 = fuzzy.fuzzy_type2()
            type2_result = f_type2.make_decision(root_exist_value, slub_exist_value, legs_exist_value)
            f_type3 = fuzzy.fuzzy_type3()
            type3_result = f_type3.make_decision(tree_size_value, trunk_size_value, nose_size_value)
            
            # db:KEYWORD에 결과 넣기
            keyword = get_keyword(h_keyword+t_keyword+p_keyword)
            # sql_keyword = f"INSERT INTO KEYWORD VALUES(\"{userid}\", \"{day}\", \"{keyword}\");"
            sql_keyword = f"UPDATE KEYWORD SET keyword_index = '{keyword}' WHERE userid = {userid} AND day = '{day}';"
            db.execute(sql_keyword)
            conn.commit()
            
            # model의 output 중 reult_report_index의 값으로 db:RAW_RESULT 값을 가져와 result라는 변수로 저장
            result_index = draw_result_index(h_output, t_output, p_output)
            
            sql_result= f"SELECT sentence FROM DRAW_RESULT;"
            db.execute(sql_result)
            draw_report = db.fetchall()

            db_result = ''
            for i in result_index:
                if draw_report[i-1][0]:
                    db_result += draw_report[i-1][0]


            # db:DRAW_REPORT에 model의 결과 넣기
            # sql_report= f"INSERT INTO DRAW_REPORT VALUES(\"{userid}\", \"{day}\", \"{h_img}\", \"{t_img}\", \"{p_img}\", \"{db_result}\", \"{h_sentence}\", \"{t_sentence}\", \"{p_sentence}\", \"{type1_result}\", \"{type2_result}\", \"{type3_result}\");"
            sql_report= f"UPDATE DRAW_REPORT SET house_img = '{h_img}', tree_img = '{t_img}', person_img = '{p_img}', result = '{db_result}', house_text = '{h_sentence}', tree_text = '{t_sentence}', person_text = '{p_sentence}', f_type1 = {type1_result}, f_type2 = {type2_result}, f_type3 = {type3_result} WHERE userid = {userid} AND day = '{day}';"
            db.execute(sql_report)
            conn.commit()
            # conn.close()

        else:
            sql_date = f"INSERT INTO USER_TESTDAY VALUES(\"{userid}\", \"{day}\");"
            db.execute(sql_date)
            conn.commit()
            # conn.close()

            # img를 서버에 /test_image 폴더에 저장하기
            h_img = 'test_images/house/image_house_' + str(day) + '.jpg'
            t_img = 'test_images/tree/image_tree_' + str(day) + '.jpg'
            p_img = 'test_images/person/image_person_' + str(day) + '.jpg'
            # h_img = 'test_images/house/reshape_test7.jpg'
            # t_img = 'test_images/person/test3.JPG'
            # p_img = 'test_images/tree/reshape_tree4.jpg'
            
            # model 실행
            h_output, h_keyword, h_sentence, house_size_value, window_size_value, chimney_value = house_output.house_print(h_img)
            t_output, t_keyword, t_sentence, root_exist_value, slub_exist_value, tree_size_value, trunk_size_value = tree_output.tree_print(t_img)
            p_output, p_keyword, p_sentence, head_size_value, nose_value, legs_exist_value, nose_size_value = person_output.person_print(p_img)

            f_type1 = fuzzy.fuzzy_type1()
            type1_result = f_type1.make_decision(house_size_value, head_size_value, nose_size_value, window_size_value, chimney_value, nose_value)
            f_type2 = fuzzy.fuzzy_type2()
            type2_result = f_type2.make_decision(root_exist_value, slub_exist_value, legs_exist_value)
            f_type3 = fuzzy.fuzzy_type3()
            type3_result = f_type3.make_decision(tree_size_value, trunk_size_value, nose_size_value)
            
            # db:KEYWORD에 결과 넣기
            keyword = get_keyword(h_keyword+t_keyword+p_keyword)
            sql_keyword = f"INSERT INTO KEYWORD VALUES(\"{userid}\", \"{day}\", \"{keyword}\");"
            db.execute(sql_keyword)
            conn.commit()
            
            # model의 output 중 reult_report_index의 값으로 db:RAW_RESULT 값을 가져와 result라는 변수로 저장
            result_index = draw_result_index(h_output, t_output, p_output)
            
            sql_result= f"SELECT sentence FROM DRAW_RESULT;"
            db.execute(sql_result)
            draw_report = db.fetchall()

            db_result = ''
            for i in result_index:
                if draw_report[i-1][0]:
                    db_result += draw_report[i-1][0]


            # db:DRAW_REPORT에 model의 결과 넣기
            sql_report= f"INSERT INTO DRAW_REPORT VALUES(\"{userid}\", \"{day}\", \"{h_img}\", \"{t_img}\", \"{p_img}\", \"{db_result}\", \"{h_sentence}\", \"{t_sentence}\", \"{p_sentence}\", \"{type1_result}\", \"{type2_result}\", \"{type3_result}\");"
            db.execute(sql_report)
            conn.commit()
            # conn.close()

    finally:
        conn.close()

    # return result
    return "HTP 보고서 생성 완료"

# @app.post("/report/{user_id}/{testday}")
# async def show_report(user_id:int, testday:date):
#     userid = user_id
#     day = testday
#     sql = f"SELECT * FROM DRAW_REPORT WHERE userid = {userid} AND day = '{testday}';"
#     db.execute(sql)
#     result = db.fetchall()
    
#     return result

@app.get("/report/keyword/{user_id}")
async def recommand_keyword(user_id: int):
    userid = user_id
    day = datetime.now().date()
    # day = '2023-02-22'
    sql_recommend = f"SELECT keyword FROM TALK_REPORT WHERE userid = {userid} AND day = '{day}';"
    db.execute(sql_recommend)
    keyword = db.fetchone()
    
    keyword_list = str(keyword[0]).split(',')
    if '8' in keyword_list:
       keyword_list.remove('8')

    result = random.choice(keyword_list)

    return {'result' : int(result)}