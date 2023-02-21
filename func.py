import numpy as np

class my_func:
  def __init__(self, detections, category_index, image_path, image_np):
    self.detections = detections
    self.category_index = category_index
    self.image_path = image_path
    self.image_np = image_np
    # house
    self.house_size_value = 0
    self.window_size_value = 0
    self.chimney_value = 0
    # tree
    self.root_exist_value = 0
    self.slub_exist_value = 0
    self.tree_size_value = 0
    self.trunk_size_value = 0
    # person
    self.head_size_value = 0
    self.nose_value = 0
    self.legs_exist_value = 0
    self.nose_size_value = 0
    # type1 = [house_size_value, head_size_value, nose_size_value, window_size_value, chimney_value, nose_value]
    # type2 = [root_exist_value, slub_exist_value, legs_exist_value]
    # type3 = [tree_size_value, trunk_size_value, nose_size_value]


  def class_cnt_list(self):
    def result_modify():
        roof_index = 100
        for i in range(3,6):
          if i in top_classes:
            find_index = np.where(top_classes == i)
            if roof_index > find_index[0][0]:
              roof_index = find_index[0][0]
          
        if top_classes[roof_index] == 3:
          result['roof2'] = 0
          result['roof3'] = 0
        elif top_classes[roof_index] == 4:
          result['roof1'] = 0
          result['roof3'] = 0
        elif top_classes[roof_index] == 5:
          result['roof1'] = 0
          result['roof2'] = 0
        return result

    score = self.detections['detection_scores'][0].numpy()
    top_score = score[score > 0.5]
    box_cnt = len(top_score)
    top_classes = self.detections['detection_classes'][0][:box_cnt].numpy()
    # print('top_score:',top_score)
    # print('top_classes:', top_classes)

    class_label = {self.category_index[i]['name']: i for i in range(1, len(self.category_index)+1)}
    # print('class_label:', class_label)

    # label_index는 n 번째  class가 있는 index와 탐지 개수 ex)[[0 2], 2]
    label_index = [] * len(self.category_index)
    for i in range(len(self.category_index)):
      label_index.append(np.where(top_classes == i))

    result = {self.category_index[i]['name'] : len(label_index[i-1][0]) for i in range(1, len(self.category_index)+1)}
    if 'roof_absence' in list(result.keys()): 
      if (result['roof1'] and result['roof2']) or (result['roof2'] and result['roof3']) or (result['roof3'] and result['roof1']):
        result_modify()

    return result, box_cnt, label_index, class_label


  def get_cordinate(self, name):
    
    result, box_cnt, label_index, class_label = self.class_cnt_list()

    global x_trans, y_trans
    tar_index = class_label[name]-1
    if result[name] == 0:
      print('이미지에서 %s을 찾을 수 없습니다.' % name)
    else:
      top_boxes = self.detections['detection_boxes'][0][:box_cnt].numpy()
      point_boxes = []
      point_boxes.append(top_boxes[label_index[tar_index][0][0]])

      xy = []
      for i in label_index[tar_index][0]:
        # 1. detections['detection_boxes'][0].numpy() 에서 name에 해당하는 값 가져오기
        point_index = self.detections['detection_boxes'][0][i].numpy()

        # 2. 좌표 분리
        x1_point = point_index[0]
        y1_point = point_index[1]
        x2_point = point_index[2]
        y2_point = point_index[3]

        # 3. input shape으로 스케일링
        x_trans = self.image_np.shape[0]
        y_trans = self.image_np.shape[1]

        new_x1_point = round(x1_point*x_trans)
        new_x2_point = round(x2_point*x_trans)

        new_y1_point = round(y1_point*y_trans)
        new_y2_point = round(y2_point*y_trans)
        
        xy.append([new_x1_point, new_y1_point, new_x2_point, new_y2_point])
      return xy

  # x_len = new_x2_point - new_x1_point
  # y_len = new_y2_point - new_y1_point

  # plt.figure(figsize=(12,16))
  # plt.imshow(image_np_with_detections)
  # rectangle = plt.Rectangle((new_y1_point,new_x1_point), y_len, x_len)

  # plt.gca().add_patch(rectangle)
  # plt.show()

  # [x1, y1, x2, y2]

  def get_location(self, name):
    xy = self.get_cordinate(name)
    lo = (xy[0][0] + xy[0][2]) / 2
    a4 = x_trans / 10

    if(lo <= a4*3):
      location = 'Top'
    elif(lo >= a4*7):
      location = 'Bottom'
    else:
      location = 'Middle'    
    return location

#----------get_size-----------#

  # 통합 ( 사람, 나무, 집 크기 )
  def get_size(self, name):
    # area = (x2-x1)*(y2-y1)
    temp = self.get_cordinate(name)
    object_area = (temp[0][2]-temp[0][0])*(temp[0][3]-temp[0][1])
    a4_area = x_trans*y_trans # a4 용지 넓이

    if (object_area/a4_area)<0.4:
      size = 'Small'
    elif (object_area/a4_area)>0.6:
      size = 'Big'
    else:
      size = 'Midium'
  
    if name == 'person':
      return size
    else:
      return size, (object_area/a4_area)*100


  # 사람
  # 머리
  # 7등신 부터 머리 작음 / 3등신부터 머리 큼
  # 키 대비 비율 ( person 세로 길이이 )
  def person_head_ratio(self):
    people = self.get_cordinate('person')
    head = self.get_cordinate('head')
    head_long = head[0][2]-head[0][0]
    people_long = people[0][2]-people[0][0]
    # people_size = size(people[0],people[1],people[2],people[3])
    if (people_long/head_long)<3:
      head_size = 'Big'
    elif (people_long/head_long)>6:
      head_size = 'Small'
    else:
      head_size = 'Midium'
    return head_size, (head_long/people_long)*100

  # 다리
  # 키 대비 비율 ( person 세로길이 )
  def person_leg_ratio(self):
    people = self.get_cordinate('person')
    leg = self.get_cordinate('legs')
    leg_long = leg[0][2]-leg[0][0]
    people_long = people[0][2]-people[0][0]
    if (people_long/leg_long)<3:
      leg_size = 'Big'
    elif (people_long/leg_long)>6:
      leg_size = 'Small'
    else:
      leg_size = 'Midium'
    return leg_size

  # 코 
  # 머리크기 대비 비율
  # x1,y1,x2,y2
  def person_nose_ratio(self):
    head = self.get_cordinate('person')
    nose = self.get_cordinate('nose')
    # head_size = (x2-x1)*(y2-y1)
    head_size = (head[0][2]-head[0][0])*(head[0][3]-head[0][1])
    nose_size = (nose[0][2]-nose[0][0])*(nose[0][3]-nose[0][1])

    if (nose_size/head_size)>0.6:
      size_nose = 'Big'
    elif (nose_size/head_size)<0.3:
      size_nose = 'Small'
    else:
      size_nose = 'Midium'
    return size_nose, (nose_size/head_size)*100

  # 나무
  # 줄기 크기
  def get_trunk_size(self):
    # 나무 가로길이
    tree = self.get_cordinate('Tree')
    trunk = self.get_cordinate('Trunk')

    tree_long = tree[0][3]-tree[0][1]
    trunk_long = trunk[0][3]-trunk[0][1]

    if (tree_long/trunk_long)<2:
      trunk_size = 'Big' #더블 첵 필요
    elif (tree_long/trunk_long)>5:
      trunk_size = 'Small' #더블 첵 필요
    else:
      trunk_size = 'Midium'
    return trunk_size, (trunk_long/tree_long)*100

  # 집
  # 창문크기
  def get_window_size(self):
    window = self.get_cordinate('window')
    house = self.get_cordinate('house')

    house_size = (house[0][2]-house[0][0])*(house[0][3]-house[0][1])
    window_size = (window[0][2]-window[0][0])*(window[0][3]-window[0][1])

    if (window_size/house_size)>0.23:
      size_window = 'Big'
    elif (window_size/house_size)<0.12:
      size_window = 'Small'
    else:
      size_window = 'Midium'
    return size_window, (window_size/house_size)*100
  
#----------Pattern-----------#

  def isPattern(self, cordinate):
    import cv2
    from PIL import Image
  
    x1, y1, x2, y2 = cordinate[0]
    #이미지 지정
    img = cv2.imread(self.image_path)

    #이미지 자르기
    cut_img = img[x1:x2, y1:y2]
    #이미지 gray화
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #원본
    gray_cut = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)   #컷팅
    #FAST 특징 검출기 생성
    fast = cv2.FastFeatureDetector_create(50)
    #특징점 개수 추출 및 출력
    keypoints = fast.detect(gray, None) #원본
    keypoints_cut = fast.detect(gray_cut, None) #컷팅

    #특징점 그리기
    cut_img_draw = cv2.drawKeypoints(cut_img, keypoints_cut, None)  
    # 결과 출력
    #cv2_imshow(cut_img_draw)
    # print(len(keypoints_cut))

    if len(keypoints_cut) > 100:
      return True
    else:
      return False

    # if len(keypoints) len(keypoints_cut) -> 기준
    # print(len(keypoints))

#-----------output------------#

  def house_output(self):

    result,_,_,_ = self.class_cnt_list()

    # {'chimney': 1, 'door': 1, 'house': 1, 'roof1': 0, 'roof2': 1, 'roof3': 0, 'window': 2}
    if not result['house']:
      return '분석이 불가능합니다. 집을 그려주세요!'
    output_label = ['chimney_absence', 'door_absence', 'house_size', 'house_location', 'roof_absence', 'roof_shape', 'roof_pattern', 'window_cnt', 'window_size']
    output = [0] * 9
    if result['chimney']:
      output[0] = 1
      self.chimeny_value = 1

    if result['door']:
      output[1] = 1

    if result['house']:
      output[2], self.house_size_value = self.get_size('house')
      output[3] = self.get_location('house')

    if result['roof1'] + result['roof2'] + result['roof3']:
      output[4] = 1
      if result['roof1']:
        output[5] = 'roof1'
      elif result['roof2']:
        output[5] = 'roof2'
      elif result['roof3']:
        output[5] = 'roof3'

      if self.isPattern(self.get_cordinate(output[5])):
        output[6] = 1

      if result['window']:
        output[8], self.window_size_value = self.get_window_size()

      if result['window']:
        output[7] = result['window']
        if result['window'] == 2:
          output[7] = 0.2
        if result['window'] >= 3:
          output[7] = 3

      output_value = dict(zip(output_label, output))
    # output: {'chimney_absence': 0, 'door_absence': 1, 'house_size': 'big', 'house_location': 'top', 'roof_absence': 1, 'roof_shape': 'roof2', 'roof_pattern': 0, 'window_cnt': 0.2, 'window_size': 'small'}
    return output_value

  
  def tree_output(self):

    result,_,_,_ = self.class_cnt_list()
    # result : {'Branch': 1, 'Fruit': 3, 'Rootdown': 0, 'Rootup': 2, 'Slub': 1, 'Tree': 1, 'Trunk': 0}
    # {'Branch': 1, 'Fruit': 2, 'Rootdown': 3, 'Rootup': 4, 'Slub': 5, 'Tree': 6, 'Trunk': 7}
    if not result['Tree']:
      return '분석이 불가능합니다. 나무를 그려주세요!'
    output_label = ['tree_size', 'branch_absence', 'fruit_absence', 'trunk_size', 'slub_absence', 'root_absence', 'root_location']
  
    output = [0] * 7
    if result['Tree']:
      output[0], self.tree_size_value = self.get_size('Tree')

    if result['Branch']:
      output[1] = 1

    if result['Fruit']:
      output[2] = 1

    if result['Trunk']:
      output[3], self.trunk_size_value = self.get_trunk_size()

    if result['Slub']:
      output[4] = 1
      self.slub_exist_value = 1

    if result['Rootup'] + result['Rootdown']:
      output[5] = 1
      self.root_exist_value = 1

      if result['Rootup']:
        output[6] = 'up'
      else:
        output[6] = 'down'


    output_value = dict(zip(output_label, output))
    return output_value

  def person_output(self):

    result,_,_,_ = self.class_cnt_list()

    if not result['person']:
      return '분석이 불가능합니다. 사람을 그려주세요!'
    output_label = ['person_size','eyes_absence','nose_absence','nose_size','mouth_shape',
                      'ears_cnt','foot_absence','legs_absence','legs_size','head_size']
    output = [0]* 13 # 기본값 = 0

    # output[0] = person_size
    if result['person']:
      output[0] = self.get_size('person')

    # output[1] = eyes_absence
    if result['eyes']:
      output[1] = 1
      
    # output[2] = nose_absence
    # output[3] = nose_size
    if result['nose']:
      output[2] = 1
      self.nose_value = 1
      output[3], self.nose_size_value = self.person_nose_ratio()
    
    # output[4] = mouth shape
    if result['mouth_open']:
      output[4] = 'mouth_open'
    if result['mouth_close']:
      output[4] = 'mouth_close'
    if result['mouth_teeth']:
      output[4] = 'mouth_teeth'

    # output[5] = ears_cnt
    if result['ears']:
      output[5] = result['ears']
    
    # output[6] = foot_absence
    if result['foot']:
      output[6] = 1 
    
    # output[7] = legs_absence
    # output[8] = legs_size
    if result['legs']:
      output[7] = 1
      self.legs_exist_value = 1
      output[8] = self.person_leg_ratio()

    # output[9] = head_size
    if result['head']:
      output[9], self.head_size_value = self.person_head_ratio()

    output_value = dict(zip(output_label, output))
    return output_value


  def sentence_print(self, type, output):
    type_key = {'house':'집','tree':'나무','person':'사람'}

    house_key = {'chimney':'굴뚝','door':'문','house':'집','roof':'지붕','window':'창문','pattern':'패턴'}
    tree_key = {'tree': '나무','branch':'가지','slub':'옹이','root':'뿌리','fruit':'열매','trunk':'나무 줄기'}
    person_key = {'person' : '사람','eyes':'눈','nose':'코','mouth':'입','foot':'발','legs':'다리','head':'머리'}

    absence_key = { 0:"없",1:'있'}
    size_key = {'Big':'큽','Midium':'중간입','Small':'작습'}
    location_key = {'Top':'상단','Middle':'중앙','Bottom':'하단','up':'상단','down':'하단'}
    shape_key = {'roof1':'1차원','roof2':'2차원','roof3':'3차원',
                 'mouth_open':'입을 벌리고 있습니다.','mouth_close':'입을 다물고 있습니다.',
                'mouth_teeth':'이빨이 보입니다.'}
    result=''
    key_list, value_list = list(output.keys()),list(output.values())
    # print(key_list)
    # print(value_list)
    # print()
    
    for i in range(len(output)):
      if value_list[i]==0:
        pass
      elif key_list[i]=='house_location' or key_list[i] == 'roof_pattern' or key_list[i]=='root_location'or key_list[i]=='nose_size' or key_list[i]=='legs_size':
          pass 
      elif type=='house':
          # size,location
          if key_list[i].split('_')[1]=="size":
              #window인 경우
              if key_list[i].split('_')[0]=='window':
                  #print('window')
                  result += '{0}의 크기가 {1}니다. \n'.format(house_key[key_list[i].split('_')[0]], size_key[output['window_size']])
              #house인 경우
              elif key_list[i].split('_')[0]=='house':
                  #print('house')
                  result += '{0}의 크기가 {1}니다. 또한, 그림은 종이의 {2}에 위치해 있습니다.\n'.format(type_key[type], size_key[output['house_size']], location_key[output['house_location']])
          # absence
          # key_list, value_list 사용
          elif key_list[i].split('_')[1]=="absence" or key_list[i].split('_')[1]== "pattern":
              result += '{0}에 {1}이 {2}습니다. \n'.format(type_key[type],house_key[key_list[i].split('_')[0]],absence_key[value_list[i]])
          
          # shape
          elif key_list[i].split('_')[1]=="shape":
              # print('shape')
              result += '{0}의 모양은 {1}입니다. \n'.format(house_key[key_list[i].split('_')[0]],shape_key[output['roof_shape']] )
          # window cnt
          elif key_list[i].split('_')[1]=="cnt":
              if output['window_cnt'] == 0.2:
                output['window_cnt'] = 2
              #print('cnt')
              result += '{0}의 개수는 {1}개 입니다. \n'.format(house_key[key_list[i].split('_')[0]],output['window_cnt'])
          else:
              print('')

      elif type =='tree':
          # size
          if key_list[i].split('_')[1]=="size":
              # tree인 경우
              if key_list[i].split('_')[0]=='tree':
                  result += '{0}의 크기가 {1}니다. \n'.format(tree_key[key_list[i].split('_')[0]], size_key[output['tree_size']])
              # trunk
              if key_list[i].split('_')[0]=='trunk':
                  result += '{0}의 크기가 {1}니다. \n'.format(tree_key[key_list[i].split('_')[0]], size_key[output['trunk_size']])                       
          #print('tree')
          # absence
          elif key_list[i].split('_')[1]=="absence":
              if key_list[i].split('_')[0]=='root':
                  result += '{0}에 {1}가 {2}고, 지면의 {3}에 위치해 있습니다. \n'.format(type_key[type],tree_key[key_list[i].split('_')[0]],absence_key[value_list[i]],location_key[output['root_location']])
              else:
                  result += '{0}에 {1}가 {2}습니다. \n'.format(type_key[type],tree_key[key_list[i].split('_')[0]],absence_key[value_list[i]])
                      
      elif type =='person':
          # size
          if key_list[i].split('_')[1]=="size":
              # tree인 경우
              if key_list[i].split('_')[0]=='person':
                  result += '{0}의 크기가 {1}니다.\n'.format(person_key[key_list[i].split('_')[0]], size_key[output['person_size']])
              if key_list[i].split('_')[0]=='legs':
                  result += '{0}의 크기가 {1}니다.\n'.format(person_key[key_list[i].split('_')[0]], size_key[output['legs_size']])   
              if key_list[i].split('_')[0]=='head':
                  result += '{0}의 크기가 {1}니다.\n'.format(person_key[key_list[i].split('_')[0]], size_key[output['head_size']])
          # absence
          elif key_list[i].split('_')[1]=="absence":
              if key_list[i].split('_')[0]=='eyes':
                  result += '{0} 그림에 {1}이 {2}습니다.\n'.format(type_key[type],person_key[key_list[i].split('_')[0]],absence_key[value_list[i]])
              elif key_list[i].split('_')[0]=='nose':
                  if output['nose_absence']==0:
                      pass
                  else:
                      result += '{0} 그림에 {1}가 {2}고, 크기가 {3}니다.\n'.format(type_key[type],person_key[key_list[i].split('_')[0]],absence_key[value_list[i]],size_key[output['nose_size']])
              elif key_list[i].split('_')[0]=='legs':
                  result += '{0} 그림에 {1}가 {2}고, 크기가 {3}니다.\n'.format(type_key[type],person_key[key_list[i].split('_')[0]],absence_key[value_list[i]],size_key[output['legs_size']]) 
          # mouth_shape         
          elif key_list[i].split('_')[1]=='shape' :
              if output['mouth_shape']==0:
                  pass
              else:
                  #print(shape_key[output['mouth_shape']])
                  result += '사람은 {0}\n'.format(shape_key[output['mouth_shape']] ) 
      else:
          result += '입력된 그림이 없습니다.'
          
    return result
    

#--------connect keyword---------#
  def house_keyword(self):
    keyword = []
    output = self.house_output()
    
    if output["chimney_absence"] == 1:
        keyword.append(0)
    
    if output["door_absence"] == 0:
        keyword.append(11)
        keyword.append(14)
    if output["house_size"] == 'Big':
        keyword.append(0)
        keyword.append(4)
  
    if output["window_size"] == 'Middle':
        keyword.append(1)
        keyword.append(8)
    elif output["window_size"] == 'Big':
        keyword.append(0)
        keyword.append(2)

    if output["house_location"] == 'Middle':
        keyword.append(3)

    if output["window_cnt"] == 1:
      keyword.append(4)
      keyword.append(7)
    elif output["window_cnt"] == 2:
      keyword.append(2)
      keyword.append(3)
      keyword.append(6)

    # keyword 리스트 중복 제거
    k_list = list(set(keyword))
    result = []
    for i in k_list:
      if keyword.count(i) > 1:
        result.append(i)

    if not len(result):
      result = k_list

    return result

  def tree_keyword(self):
    keyword = []
    keyword.append(2)

    output = self.tree_output()

    if output["root_absence"]:
      keyword.append(13)

    if output["root_location"] == 'Bottom':
      keyword.append(0)

    if output["slub_absence"]:
      keyword.append(3)
      keyword.append(6)
      keyword.append(7)
    elif output["slub_absence"] == 0:
      keyword.append(0)
      keyword.append(17)
    
    if output["fruit_absence"]:
      keyword.append(3)
      keyword.append(6)
      keyword.append(7)
    elif output["fruit_absence"] == 0:
      keyword.append(0)
      keyword.append(14)

    if output["tree_size"] == 'Small':
      keyword.append(0)
      keyword.append(5)
      keyword.append(13)
      keyword.append(14)
    elif output["tree_size"] == 'Big':
      keyword.append(0)
      keyword.append(3)
      keyword.append(14)
      keyword.append(16)

    if output["trunk_size"] == 'Small':
      keyword.append(13)
      keyword.append(14)
    elif output["trunk_size"] == 'Big':
      keyword.append(15)

    # keyword 리스트 중복 제거
    k_list = list(set(keyword))
    result = []
    for i in k_list:
      if keyword.count(i) > 1:
        result.append(i)

    if not len(result):
      result = k_list

    return result

  def person_keyword(self):
    keyword = []
    output = self.person_output()

    if output["person_size"] == 'Midium':
      keyword.append(0)
      keyword.append(14)
    elif output['person_size'] == 'Big':
      keyword.append(0)

    if output['eyes_absence'] == 0:
      keyword.append(3)
      keyword.append(5)
      keyword.append(11)

    if output['nose_absence'] == 0:
      keyword.append(5)
      keyword.append(13)

    if output['nose_size'] == 'Small':
      keyword.append(3)
      keyword.append(4)
      keyword.append(9)
      keyword.append(5)
    
    if output['mouth_shape'] == 0:
      keyword.append(3)
      keyword.append(14)
    elif output['mouth_shape'] == 'mouth_open':
      keyword.append(10)
      keyword.append(13)
    elif output['mouth_shape'] == 'mouth_teeth':
      keyword.append(3)
      keyword.append(16)
    
    if output['ears_cnt'] == 1:
      keyword.append(0)
      keyword.append(15)
    elif output['ears_cnt'] == 2:
      keyword.append(0)
      keyword.append(5)

    if output['foot_absence'] == 0:
      keyword.append(17)
      keyword.append(18)

    if output['legs_absence'] == 0:
      keyword.append(10)
      keyword.append(13)

    if output['legs_size'] == 'Small':
      keyword.append(17)
    elif output['legs_size'] == 'Big':
      keyword.append(5)

    # keyword 리스트 -> 중복된 순서로 sort
    k_list = list(set(keyword))
    result = []
    for i in k_list:
      if keyword.count(i) > 1:
        result.append(i)

    if not len(result):
      result = k_list

    return result




        
    