import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

#----------type1----------#

class fuzzy_type1:

    def __init__(self):
        self.x_house_size = np.arange(0, 101, 1) # house_size/a4_size*100
        self.x_head_size = np.arange(0, 101, 1) # head_long/person_long*100
        self.x_nose_size = np.arange(0, 101, 1) # nose_size/nose_size*100
        self.x_window_size = np.arange(0, 101, 1) # window_size/house_size*100
        self.x_chimney = np.arange(0, 2, 0.02)
        self.x_nose = np.arange(0, 2, 0.02)

        # house_size
        self.house_size_small = fuzz.trapmf(self.x_house_size, [0, 0, 30, 40])
        self.house_size_medium = fuzz.trapmf(self.x_house_size, [35, 45, 55, 65])
        self.house_size_big = fuzz.trapmf(self.x_house_size, [60, 70, 100, 100])

        # head_size
        #note: 작은기준은 7배 -> 14.27% / 큰 기준은 3배 -> 33.33%
        self.head_size_big = fuzz.trapmf(self.x_head_size, [0, 0, 15, 30])
        self.head_size_medium = fuzz.trapmf(self.x_head_size, [25, 40, 50, 65]) 
        self.head_size_small = fuzz.trapmf(self.x_head_size, [60, 75, 100, 100])

        # nose_size
        self.nose_size_small = fuzz.trapmf(self.x_nose_size, [0, 0, 15, 30])
        self.nose_size_medium = fuzz.trapmf(self.x_nose_size, [20, 40, 50, 70])
        self.nose_size_big = fuzz.trapmf(self.x_nose_size, [60, 75, 100, 100])

        # window_size
        self.window_size_small = fuzz.trapmf(self.x_window_size, [0, 0, 5, 12])
        self.window_size_medium = fuzz.trapmf(self.x_window_size, [10, 15, 20, 25])
        self.window_size_big = fuzz.trapmf(self.x_window_size, [23, 30, 100, 100])

        # chimney_exist
        self.chimney_exist_high = fuzz.trapmf(self.x_chimney, [0.8, 1.5, 2, 2])
        self.chimney_none = fuzz.trapmf(self.x_chimney, [0, 0, 0.8, 1])

        # nose_none
        self.nose_none_low = fuzz.trapmf(self.x_nose, [0, 0, 0.5, 1.2])
        self.nose_exist = fuzz.trapmf(self.x_nose, [1, 1.2, 2, 2])

        self.x_type1 = np.arange(0, 100, 1)

        # type1 output fuzzy sets
        self.type1_very_low = fuzz.trimf(self.x_type1, [0, 0, 30])
        self.type1_low = fuzz.trimf(self.x_type1, [0, 30, 60])
        self.type1_medium = fuzz.trimf(self.x_type1, [20, 50, 80])
        self.type1_high = fuzz.trimf(self.x_type1, [40, 70, 100])
        self.type1_very_high = fuzz.trimf(self.x_type1,[70, 100, 100])

    def and_rule(self, x, y, z):
        rule = np.fmin(x, y)
        act = np.fmin(rule, z)
        return act     

    def or_rule(self, x, y, z):
        rule = np.fmax(x, y)
        act = np.fmax(rule, z)
        return act

    def apply_type1_rules(self, house_size_value, head_size_value, nose_size_value, window_size_value, chimney_value, nose_value):
    
        # house size value functions
        house_size_level_low = fuzz.interp_membership(self.x_house_size, self.house_size_small, house_size_value)
        house_size_level_medium = fuzz.interp_membership(self.x_house_size, self.house_size_medium, house_size_value)
        house_size_level_high = fuzz.interp_membership(self.x_house_size, self.house_size_big, house_size_value)

        # head size value functions
        head_size_level_low = fuzz.interp_membership(self.x_head_size, self.head_size_small, head_size_value)
        head_size_level_medium = fuzz.interp_membership(self.x_head_size, self.head_size_medium, head_size_value)
        head_size_level_high = fuzz.interp_membership(self.x_head_size, self.head_size_big, head_size_value)

        # nose size value functions
        nose_size_level_low = fuzz.interp_membership(self.x_nose_size, self.nose_size_small, nose_size_value)
        nose_size_level_medium = fuzz.interp_membership(self.x_nose_size, self.nose_size_medium, nose_size_value)
        nose_size_level_high = fuzz.interp_membership(self.x_nose_size, self.nose_size_big, nose_size_value)
        
        # window size
        window_size_level_low = fuzz.interp_membership(self.x_window_size, self.window_size_small, window_size_value)
        window_size_level_medium = fuzz.interp_membership(self.x_window_size, self.window_size_medium, window_size_value)
        window_size_level_high = fuzz.interp_membership(self.x_window_size, self.window_size_big, window_size_value)
        
        # chimney_exist
        chimney_exist_level_high = fuzz.interp_membership(self.x_chimney, self.chimney_exist_high, chimney_value)
        chimney_none_level = fuzz.interp_membership(self.x_chimney, self.chimney_none, chimney_value)

        # nose_none
        nose_none_level_low = fuzz.interp_membership(self.x_nose, self.nose_none_low, nose_value)
        nose_exist_level = fuzz.interp_membership(self.x_nose, self.nose_exist, nose_value)

        
        # rules
        # Very_Low
        type1_act_very_low_step1 = self.and_rule(house_size_level_low, nose_size_level_low, self.type1_very_low)
        type1_act_very_low1 = np.fmin(type1_act_very_low_step1, head_size_level_high)
        
        type1_act_very_low_step2 = self.and_rule(window_size_level_low, chimney_none_level, self.type1_very_low)
        type1_act_very_low2 = np.fmin(type1_act_very_low_step2, nose_none_level_low)

        type1_act_very_low_step3 = self.and_rule(window_size_level_medium, chimney_none_level, self.type1_very_low)
        type1_act_very_low3 = np.fmin(type1_act_very_low_step3, nose_none_level_low)
                

        # Low
        type1_act_low_step1 = self.and_rule(house_size_level_low, nose_size_level_low, self.type1_low)
        type1_act_low1 = np.fmin(type1_act_low_step1, head_size_level_low)
        
        type1_act_low_step2 = self.and_rule(house_size_level_low, nose_size_level_low, self.type1_low)
        type1_act_low2 = np.fmin(type1_act_low_step2, head_size_level_medium)
        
        
        type1_act_low_step3 = self.and_rule(house_size_level_low, nose_size_level_medium, self.type1_low)
        type1_act_low3 = np.fmin(type1_act_low_step3, head_size_level_high)
        
        type1_act_low_step4 = self.and_rule(house_size_level_low, nose_size_level_high, self.type1_low)
        type1_act_low4 = np.fmin(type1_act_low_step4, head_size_level_high)
        
        type1_act_low_step5 = self.and_rule(house_size_level_medium, nose_size_level_low, self.type1_low)
        type1_act_low5 = np.fmin(type1_act_low_step5, head_size_level_high)
        
        type1_act_low_step6 = self.and_rule(house_size_level_high , nose_size_level_low, self.type1_low)
        type1_act_low6 = np.fmin(type1_act_low_step6, head_size_level_high)
        
        type1_act_low_step7 = self.and_rule(window_size_level_low, chimney_none_level, self.type1_low)
        type1_act_low7 = np.fmin(type1_act_low_step7, nose_exist_level)
        
        type1_act_low_step8 = self.and_rule(window_size_level_medium, chimney_none_level, self.type1_low)
        type1_act_low8 = np.fmin(type1_act_low_step8, nose_exist_level)
        
        type1_act_low_step9 = self.and_rule(window_size_level_low, chimney_exist_level_high, self.type1_low)
        type1_act_low9 = np.fmin(type1_act_low_step9, nose_none_level_low)
        
        type1_act_low_step10 = self.and_rule(window_size_level_medium, chimney_exist_level_high, self.type1_low)
        type1_act_low10 = np.fmin(type1_act_low_step10, nose_none_level_low)
        
        type1_act_low_step11 = self.and_rule(window_size_level_high, chimney_none_level, self.type1_low)
        type1_act_low11 = np.fmin(type1_act_low_step11, nose_none_level_low)

        
        # Medium
        type1_act_medium_step1 = self.and_rule(house_size_level_low, nose_size_level_medium, self.type1_medium)
        type1_act_medium1 = np.fmin(type1_act_medium_step1, head_size_level_medium)
        
        type1_act_medium_step2 = self.and_rule(house_size_level_medium , nose_size_level_low, self.type1_medium)
        type1_act_medium2 = np.fmin(type1_act_medium_step2, head_size_level_medium)
        
        type1_act_medium_step3 = self.and_rule(house_size_level_medium , nose_size_level_medium, self.type1_medium)
        type1_act_medium3 = np.fmin(type1_act_medium_step3, head_size_level_high)
        
        type1_act_medium_step4 = self.and_rule(house_size_level_low , nose_size_level_medium, self.type1_medium)
        type1_act_medium4 = np.fmin(type1_act_medium_step4, head_size_level_low)
        
        type1_act_medium_step5 = self.and_rule(house_size_level_low , nose_size_level_high, self.type1_medium)
        type1_act_medium5 = np.fmin(type1_act_medium_step5, head_size_level_medium)
        
        type1_act_medium_step6 = self.and_rule(house_size_level_medium , nose_size_level_low, self.type1_medium)
        type1_act_medium6 = np.fmin(type1_act_medium_step6, head_size_level_low)
        
        type1_act_medium_step7 = self.and_rule(house_size_level_high , nose_size_level_low, self.type1_medium)
        type1_act_medium7 = np.fmin(type1_act_medium_step7, head_size_level_medium)
        
        type1_act_medium_step8 = self.and_rule(house_size_level_medium , nose_size_level_high, self.type1_medium)
        type1_act_medium8 = np.fmin(type1_act_medium_step8, head_size_level_high)
        
        type1_act_medium_step9 = self.and_rule(house_size_level_high , nose_size_level_medium, self.type1_medium)
        type1_act_medium9 = np.fmin(type1_act_medium_step9, head_size_level_high)
        
        
        
        # High 
        type1_act_high_step1 = self.and_rule(house_size_level_medium , nose_size_level_medium, self.type1_high)
        type1_act_high1 = np.fmin(type1_act_high_step1, head_size_level_low)
        
        type1_act_high_step2 = self.and_rule(house_size_level_medium , nose_size_level_high, self.type1_high)
        type1_act_high2 = np.fmin(type1_act_high_step2, head_size_level_medium)
        
        type1_act_high_step3 = self.and_rule(house_size_level_high , nose_size_level_medium, self.type1_high)
        type1_act_high3 = np.fmin(type1_act_high_step3, head_size_level_medium)
        
        type1_act_high_step4 = self.and_rule(house_size_level_medium , nose_size_level_high, self.type1_high)
        type1_act_high4 = np.fmin(type1_act_high_step4 ,head_size_level_low)
        
        type1_act_high_step5 = self.and_rule(house_size_level_high , nose_size_level_medium, self.type1_high)
        type1_act_high5 = np.fmin(type1_act_high_step5, head_size_level_low)
        
        type1_act_high_step6 = self.and_rule(house_size_level_high , nose_size_level_high, self.type1_high)
        type1_act_high6 = np.fmin(type1_act_high_step6, head_size_level_medium)
        
        type1_act_high_step7 = self.and_rule(window_size_level_low, chimney_none_level, self.type1_high)
        type1_act_high7 = np.fmin(type1_act_high_step7, nose_exist_level)
        
        type1_act_high_step8 = self.and_rule(window_size_level_medium, chimney_none_level, self.type1_high)
        type1_act_high8 = np.fmin(type1_act_high_step8, nose_exist_level)
        
        type1_act_high_step9 = self.and_rule(window_size_level_high, chimney_none_level, self.type1_high)
        type1_act_high9 = np.fmin(type1_act_high_step9, nose_exist_level)
        
        type1_act_high_step10 = self.and_rule(window_size_level_high, chimney_exist_level_high, self.type1_high)
        type1_act_high10 = np.fmin(type1_act_high_step10, nose_none_level_low)

        # Very_High
        type1_act_very_high_step1 = self.and_rule(house_size_level_high , nose_size_level_high, self.type1_very_high)
        type1_act_very_high1 = np.fmin(type1_act_very_high_step1, head_size_level_low)
        
        type1_act_very_high_step2 = self.and_rule(window_size_level_high,chimney_exist_level_high, self.type1_very_high)
        type1_act_very_high2 = np.fmin(type1_act_very_high_step2, nose_exist_level)
        
        # combine the rules
        #very_low
        type1_act_very_low = self.or_rule(type1_act_very_low1, type1_act_very_low2, type1_act_very_low3)
        
        #low
        low_step1 = self.or_rule(type1_act_low1, type1_act_low2, type1_act_low3)
        low_step2 = self.or_rule(type1_act_low4, type1_act_low5, type1_act_low6)
        low_step3 = self.or_rule(type1_act_low7, type1_act_low8, type1_act_low9)
        low_step4 = np.fmax(type1_act_low10, type1_act_low11)
        
        low_step = self.or_rule(low_step1, low_step2, low_step3)
        type1_act_low = np.fmax(low_step, low_step4)
        
        
        #medium
        medium_step1 = self.or_rule(type1_act_medium1, type1_act_medium2, type1_act_medium3)
        medium_step2 = self.or_rule(type1_act_medium4, type1_act_medium5, type1_act_medium6)
        medium_step3 = self.or_rule(type1_act_medium7, type1_act_medium8, type1_act_medium9)
        type1_act_medium = self.or_rule(medium_step1, medium_step2, medium_step3)
        

        #high
        high_step1 = self.or_rule(type1_act_high1, type1_act_high2, type1_act_high3)
        high_step2 = self.or_rule(type1_act_high4, type1_act_high5, type1_act_high6)
        high_step3 = self.or_rule(type1_act_high7, type1_act_high8, type1_act_high9)
        high_step4 = self.or_rule(type1_act_high10, high_step1, high_step2)
        
        type1_act_high = np.fmax(high_step3, high_step4)
        
        #very_high
        type1_act_very_high = np.fmax(type1_act_very_high1, type1_act_very_high2)
        
        
        #type1
        step = self.or_rule(type1_act_very_low, type1_act_low, type1_act_medium)
        type1 = self.or_rule(step, type1_act_high, type1_act_very_high)
        
        
        # if we want to see the graph of the output
        # if verbose == 1:
        #     plt.rcParams["figure.figsize"] = 15, 4
        #     plt.plot(x_type1, type1_very_low, 'c', linestyle='--', linewidth=1.5, label='Very Low')
        #     plt.plot(x_type1, type1_low, 'b', linestyle='--', linewidth=1.5, label='Low')
        #     plt.plot(x_type1, type1_medium, 'g', linestyle='--', linewidth=1.5, label='Medium')
        #     plt.plot(x_type1, type1_high, 'r', linestyle='--', linewidth=1.5, label='High')
        #     plt.plot(x_type1, type1_very_high, 'y', linestyle='--', linewidth=1.5, label='Very High'),plt.title("type2 Evaluation Output")
        #     plt.legend()
            
        #     plt.fill_between(x_type1, type1, color='r')
        #     plt.ylim(-0.1, 1.1)
        #     plt.grid(True)
        #     plt.show()
        
        return type1 


    def make_decision(self, house_size_value, head_size_value, nose_size_value, window_size_value, chimney_value, nose_value):
        type1 = self.apply_type1_rules(house_size_value, head_size_value, nose_size_value, window_size_value, chimney_value, nose_value)
        # defuzzification with mean of maximum
        defuzz_type1 = fuzz.defuzz(self.x_type1, type1, 'centroid')
        # max_n = np.max(type1)

        # if (verbose == 1):
        #     plt.rcParams["figure.figsize"] = 15, 4
        #     plt.plot(x_type1, type1_very_low, 'c', linestyle='--', linewidth=1.5, label='Very Low')
        #     plt.plot(x_type1, type1_low, 'b', linestyle='--', linewidth=1.5, label='Low')
        #     plt.plot(x_type1, type1_medium, 'g', linestyle='--', linewidth=1.5, label='Medium')
        #     plt.plot(x_type1, type1_high, 'r', linestyle='--', linewidth=1.5, label='High')
        #     plt.plot(x_type1, type1_very_high, 'y', linestyle='--', linewidth=1.5, label='Very High'),plt.title("type1 Value")
        #     plt.legend()

        #     plt.fill_between(x_type1, type1, color='b')
        #     plt.ylim(-0.1, 1.1)
        #     plt.grid(True)

        #     plt.plot(defuzz_type1, max_n, 'X', color='r')
        #     plt.show()

        # print("Output: ", defuzz_type1)
        return defuzz_type1

#----------type2----------#

class fuzzy_type2:

    def __init__(self):
        self.x_root_exist = np.arange(0, 2, 0.02)
        self.x_slub_exist = np.arange(0, 2, 0.02)
        self.x_legs_exist = np.arange(0, 2, 0.02)

        # root exist value
        self.root_exist_high = fuzz.trapmf(self.x_root_exist, [0.5, 0.5, 1, 1])
        self.root_exist_none = fuzz.trapmf(self.x_root_exist, [0, 0, 0.5, 0.5])

        # slub exist value
        self.slub_exist_high = fuzz.trapmf(self.x_slub_exist, [0.5, 0.5, 1, 1])
        self.slub_exist_none = fuzz.trapmf(self.x_slub_exist, [0, 0, 0.5, 0.5])

        # legs exist value
        self.legs_exist_low = fuzz.trapmf(self.x_legs_exist, [0, 0, 0.5, 0.5])
        self.legs_exist_none = fuzz.trapmf(self.x_legs_exist, [0.5, 0.5, 1, 1])

        self.x_type2 = np.arange(0, 100, 1)

        # type2 evalutation output fuzzy sets
        self.type2_very_low = fuzz.trimf(self.x_type2, [0, 0, 30])
        self.type2_low = fuzz.trimf(self.x_type2, [0, 30, 60])
        self.type2_medium = fuzz.trimf(self.x_type2, [20, 50, 80])
        self.type2_high = fuzz.trimf(self.x_type2, [40, 70, 100])
        self.type2_very_high = fuzz.trimf(self.x_type2, [70, 100, 100])

    def and_rule(self, x, y, z):
        rule = np.fmin(x, y)
        act = np.fmin(rule, z)
        return act     

    def or_rule(self, x, y, z):
        rule = np.fmax(x, y)
        act = np.fmax(rule, z)
        return act

    def apply_type2_rules(self, root_exist_value, slub_exist_value, legs_exist_value):
        # root_exist
        #root_exist_level_low = fuzz.interp_membership(x_root_exist, root_exist_low, root_exist_value)
        #root_exist_level_medium = fuzz.interp_membership(x_root_exist, root_exist_medium, root_exist_value)
        root_exist_level_high  = fuzz.interp_membership(self.x_root_exist, self.root_exist_high, root_exist_value)
        root_exist_level_none = fuzz.interp_membership(self.x_root_exist, self.root_exist_none, root_exist_value)

        # slub_exist
        #slub_exist_level_low = fuzz.interp_membership(x_slub_exist, slub_exist_low, slub_exist_value)
        #slub_exist_level_medium = fuzz.interp_membership(x_slub_exist, slub_exist_medium, slub_exist_value)
        slub_exist_level_high = fuzz.interp_membership(self.x_slub_exist, self.slub_exist_high, slub_exist_value)
        slub_exist_level_none = fuzz.interp_membership(self.x_slub_exist, self.slub_exist_none, slub_exist_value)
        
        # legs_exist
        legs_exist_level_low = fuzz.interp_membership(self.x_legs_exist, self.legs_exist_low, legs_exist_value)
        legs_exist_level_none = fuzz.interp_membership(self.x_legs_exist, self.legs_exist_none, legs_exist_value)
        #legs_exist_level_medium = fuzz.interp_membership(x_legs_exist, legs_exist_medium, legs_exist_value)
        #legs_exist_level_high = fuzz.interp_membership(x_legs_exist, legs_exist_high, legs_exist_value)
        
        ### rules
        # 1. If (legs_exist is Low) then (type2 is Very_Low)
        type2_act_very_low1 = np.fmin(legs_exist_level_low, self.type2_very_low)
        # 2. If (root_exist is High) and (legs_exist is Low) then (type2 is Low)
        type2_act_low1 = self.and_rule(root_exist_level_high, legs_exist_level_low, self.type2_low)
        # 3. If (slub_exist is High) and (legs_exist is Low) then (type2 is Low)
        type2_act_low2 = self.and_rule(slub_exist_level_high, legs_exist_level_low, self.type2_low)
        # 4. If (root_exist is High) then (type2 is Medium)
        type2_act_medium1 = np.fmin(root_exist_level_high, self.type2_medium)
        # 5. If (slub_exist is High) then (type2 is Medium)
        type2_act_medium2 = np.fmin(slub_exist_level_high, self.type2_medium)
        # 6. If (root_exist is High) and (slub_exist is High) and (legs_exist is Low) then (type2 is High)
        type2_act_high_step = self.and_rule(root_exist_level_high, slub_exist_level_high, self.type2_high)
        type2_act_high = np.fmin(type2_act_high_step, legs_exist_level_low)
        # 7. If (root_exist is High) and (slub_exist High) then (type2 is Very_High)
        type2_act_very_high = self.and_rule(root_exist_level_high, slub_exist_level_high, self.type2_very_high)
         # 8. If ( root,slub, legs == None ) then type2 == Very_Low 
        type2_act_very_low2 = self.and_rule(root_exist_level_none, slub_exist_level_none, legs_exist_level_none)
        
        
        # combine the rules
        type2_act_low = np.fmax(type2_act_low1, type2_act_low2)
        type2_act_medium = np.fmax(type2_act_medium1, type2_act_medium2)
        type2_act_very_low = np.fmax(type2_act_very_low1, type2_act_very_low2)
    
        step = self.or_rule(type2_act_very_low, type2_act_low, type2_act_medium)
        type2 = self.or_rule(step, type2_act_high, type2_act_very_high)
        
        # if we want to see the graph of the output
        # if verbose == 1:
        #     plt.rcParams["figure.figsize"] = 15, 4
        #     plt.plot(x_type2, type2_very_low, 'c', linestyle='--', linewidth=1.5, label='Very Low')
        #     plt.plot(x_type2, type2_low, 'b', linestyle='--', linewidth=1.5, label='Low')
        #     plt.plot(x_type2, type2_medium, 'g', linestyle='--', linewidth=1.5, label='Medium')
        #     plt.plot(x_type2, type2_high, 'r', linestyle='--', linewidth=1.5, label='High')
        #     plt.plot(x_type2, type2_very_high, 'y', linestyle='--', linewidth=1.5, label='Very High'),plt.title("type2 Evaluation Output")
        #     plt.legend()
            
        #     plt.fill_between(x_type2, type2, color='r')
        #     plt.ylim(-0.1, 1.1)
        #     plt.grid(True)
        #     plt.show()
        
        return type2


    def make_decision(self, root_exist_value, slub_exist_value, legs_exist_value):
        type2 = self.apply_type2_rules(root_exist_value, slub_exist_value, legs_exist_value)
        # defuzzification with mean of maximum
        defuzz_type2 = fuzz.defuzz(self.x_type2, type2, 'centroid') # 무게중심법(centroid) 사용
        # max_n = np.max(type2)

        # if (verbose == 1):
        #     plt.rcParams["figure.figsize"] = 15, 4
        #     plt.plot(x_type2, type2_very_low, 'c', linestyle='--', linewidth=1.5, label='Very Low')
        #     plt.plot(x_type2, type2_low, 'b', linestyle='--', linewidth=1.5, label='Low')
        #     plt.plot(x_type2, type2_medium, 'g', linestyle='--', linewidth=1.5, label='Medium')
        #     plt.plot(x_type2, type2_high, 'r', linestyle='--', linewidth=1.5, label='High')
        #     plt.plot(x_type2, type2_very_high, 'y', linestyle='--', linewidth=1.5, label='Very High'),plt.title("type2 Value")
        #     plt.legend()

        #     plt.fill_between(x_type2, type2, color='b')
        #     plt.ylim(-0.1, 1.1)
        #     plt.grid(True)

        #     plt.plot(defuzz_type2, max_n, 'X', color='r')
        #     plt.show()

        # print("Output: ", defuzz_type2)
        return defuzz_type2

#----------type3----------#
                    
class fuzzy_type3:

    def __init__(self):
        self.x_tree_size = np.arange(0, 101, 1)
        self.x_trunk_size = np. arange(0, 101, 1)
        self.x_nose_size = np.arange(0, 101, 1)

        # tree size value
        self.tree_size_low = fuzz.trapmf(self.x_tree_size, [0, 0, 25, 40])
        self.tree_size_medium = fuzz.trapmf(self.x_tree_size, [30, 45, 55, 70])
        self.tree_size_high = fuzz.trapmf(self.x_tree_size, [60, 75, 100, 100])

        # trunk size value
        self.trunk_size_low = fuzz.trapmf(self.x_trunk_size, [0, 0, 10, 20])
        self.trunk_size_medium = fuzz.trapmf(self.x_trunk_size, [10, 30, 40, 60])
        self.trunk_size_high = fuzz.trapmf(self.x_trunk_size, [50, 60, 100, 100])

        # tree size value
        self.nose_size_low = fuzz.trapmf(self.x_nose_size, [0, 0, 15, 30])
        self.nose_size_medium = fuzz.trapmf(self.x_nose_size, [20, 40, 50, 70])
        self.nose_size_high = fuzz.trapmf(self.x_nose_size, [60, 75, 100, 100])

        self.x_type3 = np.arange(0, 100, 1)

        # type3 evalutation output fuzzy sets
        self.type3_very_low = fuzz.trimf(self.x_type3, [0, 0, 30])
        self.type3_low = fuzz.trimf(self.x_type3, [0, 30, 60])
        self.type3_medium = fuzz.trimf(self.x_type3, [20, 50, 80])
        self.type3_high = fuzz.trimf(self.x_type3, [40, 70, 100])
        self.type3_very_high = fuzz.trimf(self.x_type3, [70, 100, 100])

    def and_rule(self, x, y, z):
        rule = np.fmin(x, y)
        act = np.fmin(rule, z)
        return act     

    def or_rule(self, x, y, z):
        rule = np.fmax(x, y)
        act = np.fmax(rule, z)
        return act

    def apply_type3_rules(self, tree_size_value, trunk_size_value, nose_size_value):
        # tree_size
        tree_size_level_low = fuzz.interp_membership(self.x_tree_size, self.tree_size_low, tree_size_value)
        tree_size_level_medium = fuzz.interp_membership(self.x_tree_size, self.tree_size_medium, tree_size_value)
        tree_size_level_high  = fuzz.interp_membership(self.x_tree_size, self.tree_size_high, tree_size_value)

        # trunk_size
        trunk_size_level_low = fuzz.interp_membership(self.x_trunk_size, self.trunk_size_low, trunk_size_value)
        trunk_size_level_medium = fuzz.interp_membership(self.x_trunk_size, self.trunk_size_medium, trunk_size_value)
        trunk_size_level_high = fuzz.interp_membership(self.x_trunk_size, self.trunk_size_high, trunk_size_value)
        
        # nose_size
        nose_size_level_low = fuzz.interp_membership(self.x_nose_size, self.nose_size_low, nose_size_value)
        nose_size_level_medium = fuzz.interp_membership(self.x_nose_size, self.nose_size_medium, nose_size_value)
        nose_size_level_high = fuzz.interp_membership(self.x_nose_size, self.nose_size_high, nose_size_value)
        
        ### rules
        # 1. If (tree_size is Low) and (trunk_size is Low) and (nose_size is High) then (type3 is Very_low)
        type3_act_very_low_step = self.and_rule(tree_size_level_low, trunk_size_level_low, self.type3_very_low)
        type3_act_very_low = np.fmin(type3_act_very_low_step, nose_size_level_high)
        # 2. If (tree_size is Low) and (trunk_size is Low) and (nose_size is not High) then (type3 is Low)
        type3_act_low1_step = self.and_rule(tree_size_level_low, trunk_size_level_low, self.type3_low)
        type3_act_low1 = np.fmin(type3_act_low1_step, nose_size_level_low)
        type3_act_low2_step = self.and_rule(tree_size_level_low, trunk_size_level_low, self.type3_low)
        type3_act_low2 = np.fmin(type3_act_low2_step, nose_size_level_medium)
        # 3. If (tree_size is Low) and (trunk_size is not Low) and (nose_size is High) then (type3 is Low)
        type3_act_low3_step = self.and_rule(tree_size_level_low, trunk_size_level_medium, self.type3_low)
        type3_act_low3 = np.fmin(type3_act_low3_step, nose_size_level_high)
        type3_act_low4_step = self.and_rule(tree_size_level_low, trunk_size_level_high, self.type3_low)
        type3_act_low4 = np.fmin(type3_act_low4_step, nose_size_level_high)
        # 4. If (tree_size is not Low) and (trunk_size is Low) and (nose_size is High) then (type3 is Low)
        type3_act_low5_step = self.and_rule(tree_size_level_medium, trunk_size_level_low, self.type3_low)
        type3_act_low5 = np.fmin(type3_act_low5_step, nose_size_level_high)
        type3_act_low6_step = self.and_rule(tree_size_level_high , trunk_size_level_low, self.type3_low)
        type3_act_low6 = np.fmin(type3_act_low6_step, nose_size_level_high)
        # 5. If (tree_size is Low) and (trunk_size is Medium) and (nose_size is Medium) then (type3 is Medium)
        type3_act_medium1_step = self.and_rule(tree_size_level_low, trunk_size_level_medium, self.type3_medium)
        type3_act_medium1 = np.fmin(type3_act_medium1_step, nose_size_level_medium)
        # 6. If (tree_size is Medium) and (trunk_size is Low) and (nose_size is Medium) then (type3 is Medium)
        type3_act_medium2_step = self.and_rule(tree_size_level_medium , trunk_size_level_low, self.type3_medium)
        type3_act_medium2 = np.fmin(type3_act_medium2_step, nose_size_level_medium)
        # 7. If (tree_size is Medium) and (trunk_size is Medium) and (nose_size is High) then (type3 is Medium)
        type3_act_medium3_step = self.and_rule(tree_size_level_medium , trunk_size_level_medium, self.type3_medium)
        type3_act_medium3 = np.fmin(type3_act_medium3_step, nose_size_level_high)
        # 8. If (tree_size is Low) and (trunk_size is Medium) and (nose_size is Low) then (type3 is Medium)
        type3_act_medium4_step = self.and_rule(tree_size_level_low , trunk_size_level_medium, self.type3_medium)
        type3_act_medium4 = np.fmin(type3_act_medium4_step, nose_size_level_low)
        # 9. If (tree_size is Low) and (trunk_size is High) and (nose_size is Medium) then (type3 is Medium)
        type3_act_medium5_step = self.and_rule(tree_size_level_low , trunk_size_level_high, self.type3_medium)
        type3_act_medium5 = np.fmin(type3_act_medium5_step, nose_size_level_medium)
        # 10. If (tree_size is Medium) and (trunk_size is Low) and (nose_size is Low) then (type3 is Medium)
        type3_act_medium6_step = self.and_rule(tree_size_level_medium , trunk_size_level_low, self.type3_medium)
        type3_act_medium6 = np.fmin(type3_act_medium6_step, nose_size_level_low)
        # 11. If (tree_size is High) and (trunk_size is Low) and (nose_size is Medium) then (type3 is Medium)
        type3_act_medium7_step = self.and_rule(tree_size_level_high , trunk_size_level_low, self.type3_medium)
        type3_act_medium7 = np.fmin(type3_act_medium7_step, nose_size_level_medium)
        # 12. If (tree_size is Medium) and (trunk_size is High) and (nose_size is High) then (type3 is Medium)
        type3_act_medium8_step = self.and_rule(tree_size_level_medium , trunk_size_level_high, self.type3_medium)
        type3_act_medium8 = np.fmin(type3_act_medium8_step, nose_size_level_high)
        # 13. If (tree_size is High) and (trunk_size is Medium) and (nose_size is High) then (type3 is Medium)
        type3_act_medium9_step = self.and_rule(tree_size_level_high , trunk_size_level_medium, self.type3_medium)
        type3_act_medium9 = np.fmin(type3_act_medium9_step, nose_size_level_high)
        # 14. If (tree_size is Medium) and (trunk_size is Medium) and (nose_size is Low) then (type3 is High)
        type3_act_high1_step = self.and_rule(tree_size_level_medium , trunk_size_level_medium, self.type3_high)
        type3_act_high1 = np.fmin(type3_act_high1_step, nose_size_level_low)
        # 15. If (tree_size is Medium) and (trunk_size is High) and (nose_size is Medium) then (type3 is High)
        type3_act_high2_step = self.and_rule(tree_size_level_medium , trunk_size_level_high, self.type3_high)
        type3_act_high2 = np.fmin(type3_act_high2_step, nose_size_level_medium)
        # 16. If (tree_size is High) and (trunk_size is Medium) and (nose_size is Medium) then (type3 is High)
        type3_act_high3_step = self.and_rule(tree_size_level_high , trunk_size_level_medium, self.type3_high)
        type3_act_high3 = np.fmin(type3_act_high3_step, nose_size_level_medium)
        # 17. If (tree_size is Medium) and (trunk_size is High) and (nose_size is Low) then (type3 is High)
        type3_act_high4_step = self.and_rule(tree_size_level_medium , trunk_size_level_high, self.type3_high)
        type3_act_high4 = np.fmin(type3_act_high4_step ,nose_size_level_low)
        # 18. If (tree_size is High) and (trunk_size is Medium) and (nose_size is Low) then (type3 is High)
        type3_act_high5_step = self.and_rule(tree_size_level_high , trunk_size_level_medium, self.type3_high)
        type3_act_high5 = np.fmin(type3_act_high5_step, nose_size_level_low)
        # 19. If (tree_size is High) and (trunk_size is High) and (nose_size is Medium) then (type3 is High)
        type3_act_high6_step = self.and_rule(tree_size_level_high , trunk_size_level_high, self.type3_high)
        type3_act_high6 = np.fmin(type3_act_high6_step, nose_size_level_medium)
        # 20. If (tree_size is High) and (trunk_size is low) and (nose_size is low) then (type3 is High)
        type3_act_high7_step = self.and_rule(tree_size_level_high , trunk_size_level_low, self.type3_high)
        type3_act_high7 = np.fmin(type3_act_high7_step, nose_size_level_low)
        # 21. If (tree_size is High) and (trunk_size is High) and (nose_size is Low) then (type3 is Very_high)
        type3_act_very_high_step = self.and_rule(tree_size_level_high , trunk_size_level_high, self.type3_very_high)
        type3_act_very_high = np.fmin(type3_act_very_high_step, nose_size_level_low)
        
        
        # combine the rules
        low_step1 = self.or_rule(type3_act_low1, type3_act_low2, type3_act_low3)
        low_step2 = self.or_rule(type3_act_low4, type3_act_low5, type3_act_low6)
        type3_act_low = np.fmax(low_step1, low_step2)
        
        medium_step1 = self.or_rule(type3_act_medium1, type3_act_medium2, type3_act_medium3)
        medium_step2 = self.or_rule(type3_act_medium4, type3_act_medium5, type3_act_medium6)
        medium_step3 = self.or_rule(type3_act_medium7, type3_act_medium8, type3_act_medium9)
        type3_act_medium = self.or_rule(medium_step1, medium_step2, medium_step3)

        high_step1 = self.or_rule(type3_act_high1, type3_act_high2, type3_act_high3)
        high_step2 = self.or_rule(type3_act_high4, type3_act_high5, type3_act_high6)
        type3_act_high = self.or_rule(type3_act_high7, high_step1, high_step2)
        
        
        step = self.or_rule(type3_act_very_low, type3_act_low, type3_act_medium)
        type3 = self.or_rule(step, type3_act_high, type3_act_very_high)
        
        # if we want to see the graph of the output
        # if verbose == 1:
        #     plt.rcParams["figure.figsize"] = 15, 4
        #     plt.plot(x_type3, type3_very_low, 'c', linestyle='--', linewidth=1.5, label='Very Low')
        #     plt.plot(x_type3, type3_low, 'b', linestyle='--', linewidth=1.5, label='Low')
        #     plt.plot(x_type3, type3_medium, 'g', linestyle='--', linewidth=1.5, label='Medium')
        #     plt.plot(x_type3, type3_high, 'r', linestyle='--', linewidth=1.5, label='High')
        #     plt.plot(x_type3, type3_very_high, 'y', linestyle='--', linewidth=1.5, label='Very High'),plt.title("Type3 Evaluation Output")
        #     plt.legend()
            
        #     plt.fill_between(x_type3, type3, color='r')
        #     plt.ylim(-0.1, 1.1)
        #     plt.grid(True)
        #     plt.show()
        
        return type3


    def make_decision(self, tree_size_value, trunk_size_value, nose_size_value):
        type3 = self.apply_type3_rules(tree_size_value, trunk_size_value, nose_size_value)
        # defuzzification with mean of maximum
        defuzz_type3 = fuzz.defuzz(self.x_type3, type3, 'centroid')
        # max_n = np.max(type3)

        # if (verbose == 1):
        #     plt.rcParams["figure.figsize"] = 15, 4
        #     plt.plot(x_type3, type3_very_low, 'c', linestyle='--', linewidth=1.5, label='Very Low')
        #     plt.plot(x_type3, type3_low, 'b', linestyle='--', linewidth=1.5, label='Low')
        #     plt.plot(x_type3, type3_medium, 'g', linestyle='--', linewidth=1.5, label='Medium')
        #     plt.plot(x_type3, type3_high, 'r', linestyle='--', linewidth=1.5, label='High')
        #     plt.plot(x_type3, type3_very_high, 'y', linestyle='--', linewidth=1.5, label='Very High'),plt.title("Type3 Value")
        #     plt.legend()

        #     plt.fill_between(x_type3, type3, color='b')
        #     plt.ylim(-0.1, 1.1)
        #     plt.grid(True)

        #     plt.plot(defuzz_type3, max_n, 'X', color='r')
        #     plt.show()s

        # print("Output: ", defuzz_type3)
        return defuzz_type3            