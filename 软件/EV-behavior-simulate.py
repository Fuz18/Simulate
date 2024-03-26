# %% [markdown]
# ### 导入库

# %%
import os  # 用于访问操作系统功能，如更改工作目录
import sys  # 用于访问与Python解释器紧密相关的变量和函数
import time as T  # 导入时间模块，用于性能测量等

import numpy as np  # 强大的数学库，提供多维数组对象及相关操作
import openpyxl as op  # 用于读写Excel 2010 xlsx/xlsm/xltx/xltm文件
import pandas as pd  # 数据处理和分析库
import scipy.optimize  # 提供常用的优化算法


# %% [markdown]
# ### 全局变量初始化

# %%
os.chdir(sys.path[0])  # 更改当前工作目录到脚本所在的目录

# %%
#该函数用于处理字符串，将其转换为数据列表。主要用于处理从文件中读取的数据行。
def split_string_to_data(string):
    string = string.replace('\n', '') # delete tail '\n'
    string = string.replace(',', ' ') # replace ',' by ' '
    string = string.replace('\t', ' ') # replace '\t' by ' '
    string = string.replace(';', ' ') # replace ';' by ' '
    while '  ' in string:
        # replace multiple spaces by one space
        string = string.replace('  ', ' ')
    # split with delimiter ' ' and store them in a list
    var_list = string.split(' ')
    while '' in var_list:
        # remove empty strings from the list
        var_list.remove('')
    return var_list

# %%
#此函数用于初始化电动汽车的一些基础参数。
def initiliaze_electric_cars(number_of_cars, EV_fleet_num):
    """
    初始化电动汽车的参数。
    :param number_of_cars: 汽车总数。
    :param EV_fleet_num: 各类电动汽车的数量。
    :return: 初始化后的汽车参数矩阵。
    """
    # 创建一个number_of_cars x 4的零矩阵，用于存储汽车参数
    cars = np.zeros((number_of_cars, 4))
    for i in range(number_of_cars):
        # 第2、3列初始化为0，分别代表每辆车的离开和到达位置
        cars[i, 1] = 0  # 离开位置
        cars[i, 2] = 0  # 到达位置
        cars[i, 3] = 1  # 初始化电池的SOC（State of Charge）为1，即100%
    t = 0
    car_num = 0
    for x in EV_fleet_num:
        # 根据EV_fleet_num设置车辆的类型
        for i in range(x):
            cars[i + car_num, 0] = t  # 车辆类型
        car_num += x
        t += 1
    return cars


# %%
# 定义时间步长和模拟天数
step = 0.25  # 时间步长（小时）
num_of_days = 30  # 模拟的天数
extra_days = 31  # 额外的天数，可能用于特殊处理或预热期

# 光伏装机功率
installed_power = 200

#【全局变量】标准Li-ion cell_18650参数定义
E0 = 3.93666 # 电池恒定电压（V）
R = 0.04 # 电池内阻（欧姆）
K = 0.0076 # 电池极化系数（V/Ah）或极化电阻（欧姆）
q = 2.2 # 电池最大容量（Ah）
q_nom = 2.1 # 电池标称容量（Ah）
A = 0.26422 # 电压模型参数
B = 26.5487 # 电压模型参数

# 车队数量和类型定义
EV_fleet_num = np.array([4, 4, 4, 4, 4]) # 各类型电动车的数量
number_of_cars = int(np.linalg.norm(EV_fleet_num, 1)) # 计算车辆总数
type = len(EV_fleet_num) # 车型数量

# 电池包配置
n_s = [81, 90, 99, 117, 135] # 串联电池单元数量
n_p = [58, 65, 69, 70, 70] # 并联电池单元数量

# 车辆重量定义
G = [11036.61, 12149.04, 13096.35, 14408.67, 15625.4] # 各类车的重量（牛顿）

# 电池性能计算
Q_max = [] # 电池最大容量列表
u = [] # 电池单元电压列表
E_max = [] # 电池最大电能列表

# 充放电倍率和电流计算
n = 0.1 # 充放电倍率C，1/h
i = n * q
for j in range(5):
    Q_max.append(q * n_p[j]) # 计算每类车的总容量
    # 根据电压模型计算标称电能
    for k in range(100):
        if k >= 94:
            u.append(E0 - R * 0.01 * q + A)
        else:
            u.append(E0 - R * i - K / ((k + 1) / 100) * (q * (1 - ((k + 1) / 100)) + i) + A * np.exp(-B * q * (1 - ((k + 1) / 100))))
    E_max.append(Q_max[j] * np.linalg.norm(u, 1) / len(u) * n_s[j] / 1000) # 计算功率（kW）

# 初始化电力使用信息
eletricity_utilization_info = np.zeros((type, 3)) # 初始化电池容量、电力消耗等信息的数组
min_power_ratio = 0.2 # 定义最低电量比率

for i in range(type):
    eletricity_utilization_info[i, 0] = n_s[i] # 电池串联数
    eletricity_utilization_info[i, 1] = n_p[i] # 电池并联数
    eletricity_utilization_info[i, 2] = G[i] # 车重

# %%
# 初始化车辆状态和距离
cars = initiliaze_electric_cars(number_of_cars, EV_fleet_num) # 假设存在初始化函数

# 定义每辆车从家到办公室的距离
home_to_office_distance = np.random.gamma(1.13, 20.29646, number_of_cars) 

# 定义积极外出的系数
k_after_work_h_o = np.random.gamma(1.1, 0.909091, number_of_cars) # 下班后从家出发的系数
k_after_work_w_o = np.random.gamma(1.4, 0.714286, number_of_cars) # 下班后从办公室出发的系数
k_weekend_h_o = np.random.gamma(1.81, 1.234568, number_of_cars) # 周末从家出发的系数

# 定义上班期间外出和休闲时间外出的比例
percentage_to_go_out_during_working = 0.475
percentage_to_go_out_leisure_time = 0.95
percentage_to_go_out_leisure_time_P1 = 0.842


# %% [markdown]
# #### 统计数据初始化

# %%
#根据日期生成车速分布
def generate_speed_distribution(day, number_of_cars):
    # day: 当前天数
    # number_of_cars: 车辆数量
    speed = week_speed[:, (day+5)%7]  # 基于星期几选择相应的速度数据
    result = np.zeros((int(24/step), number_of_cars))
    for i, x in np.ndenumerate(speed):
        # 以x为均值，2为标准差生成正态分布的速度
        temp = np.random.normal(x, 2, number_of_cars)
        for k in range(len(temp)):
            if temp[k] <= 10:
                temp[k] = x  # 速度太低时使用基准速度
            result[i, k] = temp[k]
    return result

# %%
# 读取光伏、行驶速度文件，并简单处理
# 初始化每周各时段的速度数据
week_speed = np.zeros((int(24/step), 7))

# 读取12月的实际光伏数据
# 初始化额外天数的电力供应数组
extra_days_power_supply = np.zeros((31, int(24/step)))
file = open("December_real.csv", 'r') #Select the month
for line_index, line_str in enumerate(file):
    for i in range(int(24 / step)):
        line_list = split_string_to_data(line_str)
        extra_days_power_supply[line_index, i] = float(line_list[i])
file.close()

# 初始化最大电力供应数组
max_power_supply = np.zeros((366, int(24/step)))
max_power = 0  # 记录最大电力供应值

# 读取2020年的实际光伏数据
max_power_supply = np.zeros((366, int(24/step)))
file = open("2020_real.csv", 'r') #Select the month
for line_index, line_str in enumerate(file):
    for i in range(int(24/step)):
        line_list = split_string_to_data(line_str)
        max_power_supply[line_index, i] = float(line_list[i])
        if max_power_supply[line_index, i] > max_power:
            max_power = max_power_supply[line_index, i]
file.close()

# 将电力数据归一化
#installed_power = 160
extra_days_power_supply *= (installed_power / max_power)
max_power_supply *= (installed_power / max_power)

# 读取北京的速度数据
week_speed = np.zeros((int(24/step), 7))
file = open("speed_Beijing.csv", "r", encoding='UTF-8') 
for line_index, line_str in enumerate(file):
    if line_index < 1:
        continue
    line_list = split_string_to_data(line_str)
    for i in range(7):
        week_speed[line_index-1, i] = float(line_list[i+1])
file.close()

# 生成车辆的今天、明天和昨天的速度分布
today_speed = generate_speed_distribution(1, number_of_cars)
tomorrow_speed = today_speed
yesterday_speed = today_speed

# 初始化车辆的到达时间、离开时间和目的地
arriving_time = -np.ones(number_of_cars)
leaving_time = -np.ones(number_of_cars)
destination = np.zeros(number_of_cars)

# 记录每辆车的电量
electricity_record = np.copy(cars[:, 3]).transpose()
# 记录每辆车是否到达过办公室
have_been_to_the_office = np.zeros(number_of_cars)


# %%
# 一堆需要统计的数据
# 行驶距离统计
traveling_distance_weekday = np.zeros((8, number_of_cars))  # 每辆车在工作日的总行驶距离
traveling_home_to_office_weekday = np.zeros((3, number_of_cars))  # 每辆车在工作日的上下班总行驶距离
traveling_distance_weekend = np.zeros((3, number_of_cars))  # 每辆车在周末的总行驶距离
traveling_home_to_office_weekend = np.zeros((3, number_of_cars))  # 每辆车在周末的上下班总行驶距离

# 电力、行为相关统计
total_power_generation = np.zeros(1)  # 总发电量
total_power_used = np.zeros(1)  # 总用电量
cars_at_office = []  # 办公室停车的车辆数统计
first_month_cars_at_office = []  # 第一个月办公室停车的车辆数统计
cars_electricity = []  # 每辆车的电量统计
first_month_cars_electricity = []  # 第一个月每辆车的电量统计
strategy = []  # 充电策略统计
discharge_strategy = []  # 放电策略统计
length_of_stay_at_office = []  # 办公室停留时间统计
building_power_consumption = []  # 建筑总用电量
power_consumption_rate = []  # 用电率
position = []  # 车辆位置统计
home_time = []  # 在家时间统计
office_time = []  # 在办公室时间统计
other_time = []  # 在其他地点时间统计
on_the_way_time = []  # 在途时间统计
voltage = []  # 电压统计
min_soc = []  # 最小电量比例
short_soc = []  # 电量不足车辆数统计
strategy_special = np.zeros((number_of_cars, int(24 / step)))  # 特殊策略

# 时间和操作统计
charging_time = np.zeros(num_of_days + extra_days)  # 充电时间
charge_time = np.zeros(num_of_days + extra_days)  # 充电时间
volt_time = np.zeros(num_of_days + extra_days)  # 电压调整时间
bi_time = np.zeros(num_of_days + extra_days)  # 二分法求解时间
total_time = np.zeros(num_of_days + extra_days)  # 总时间
charge_num = np.zeros(num_of_days + extra_days)  # 充电次数

# 行为统计
night_record = []  # 夜间记录
morning = [0]  # 早上外出次数
noon = [0]  # 中午外出次数
afternoon_out = [0]  # 下午外出次数
evening1 = [0]  # 傍晚1外出次数
evening2 = [0]  # 傍晚2外出次数
commuting = np.zeros((number_of_cars, num_of_days))  # 上下班通勤次数


# %% [markdown]
# ### 各种充放电相关函数

# %%
#此函数定义了充电策略，包括如何计算充电功率和电压等。
def charging_strategy(pre_location, pre_arriving, day, time):
    #声明为全局变量，以便外部调用
    global Pc_c_max  # 充电功率最大值
    global Pc_d_max  # 放电功率最大值（在此模型中不考虑放电）
    global U_d_max   # 电池放电最大电压
    global U_mid0    # 死区中心电压
    global U_c_max   # 电池充电最大电压
    global dU        # 死区电压范围
    global k1        # 死区中心电压的SOC影响因子
    global k2        # 充放电功率曲线的SOC影响因子
    global Pv_d_max  # 放电功率最小值
    global h         # 充电效率
    # 初始化充电策略的相关参数（每次都记得改一下这里）
    Pc_c_max = 10  # 充电功率最大值
    Pc_d_max = 0   # 放电功率最大值（在此模型中不考虑放电）
    U_d_max = 310  # 电池放电最大电压
    U_mid0 = 350   # 死区中心电压
    U_c_max = 390  # 电池充电最大电压
    dU = 0         # 死区电压范围
    k1 = 0         # 死区中心电压的SOC影响因子
    k2 = 0         # 充放电功率曲线的SOC影响因子
    Pv_d_max = -20 # 放电功率最小值
    h = 0.94       # 充电效率
    
    #### 内嵌函数定义 ####

    # 计算充电或放电功率
    def power_calculation(Pc_c_max, Pc_d_max, Pv_c_max, Pv_d_max, U_d_max, U_mid0, U_c_max, dU, k1, k2, U, SOC):
        # 计算中间电压，取决于SOC和电压参数
        U_mid = (1 - k1) * U_mid0 + k1 * (SOC * (U_c_max - U_d_max - 2 * dU) + U_d_max + dU)
        # 避免SOC为1或0时的计算错误
        if SOC == 1:
            SOC = 0.99999999
        elif SOC <= 0:
            SOC = 0.00000001

        # 计算充电和放电时的效率变化
        nc = (1 - k2) + k2 * SOC / (1 - SOC)
        nd = (1 - k2) + k2 * (1 - SOC) / SOC

        # 根据电压和SOC确定充电或放电功率
        if U < U_d_max:
            P = Pc_d_max
        elif U < U_mid - dU:
            P = Pc_d_max * ((U_mid - dU - U) / (U_mid - dU - U_d_max)) ** nd
        elif U < U_mid + dU:
            P = 2  # 在电压死区中设定一个固定功率值
        elif U < U_c_max:
            P = Pc_c_max * ((U - U_mid - dU) / (U_c_max - U_mid - dU)) ** nc
        else:
            P = Pc_c_max

        # 确保功率不超过最大或最小限制
        P = min(P, Pv_c_max)
        P = max(P, Pv_d_max)
        return P

    # 计算电压差异以调整充电策略
    def volt_calculation(volt, number, pv, soc, co_charge, Pv_c_max):
        t1 = T.time_ns()
        p = np.zeros(number)  # 初始化p-U曲线决定充电功率
        p_charge_cal = 0
        for i in range(number):
            p[i] = co_charge[i] * power_calculation(Pc_c_max, Pc_d_max, Pv_c_max[i], Pv_d_max, U_d_max, U_mid0, U_c_max, dU,
                                                    k1, k2, volt, soc[i]) / h
            p_charge_cal += p[i]
        difference = p_charge_cal - pv  # 计算总充电功率与供电能力的差异
        t2 = T.time_ns()
        volt_time[day-1] += (t2-t1)/1e9
        return difference

    # 使用二分法解决电压调整问题
    def bisection_solve(f, a, b, number, pv, soc, co_charge, Pv_c_max):
        t1 = T.time_ns()
        epsilon = 0.0001  # 定义精度
        c = (a + b) / 2  # 初始化中值
        while abs(f(c, number, pv, soc, co_charge, Pv_c_max)) > epsilon and abs(a-b) > 1e-10:
            c = (a + b) / 2
            if f(c, number, pv, soc, co_charge, Pv_c_max) > epsilon:
                a = c
            elif f(c, number, pv, soc, co_charge, Pv_c_max) < -epsilon:
                b = c
        t2 = T.time_ns()
        bi_time[day-1] += (t2-t1)/1e9
        return b

    #### 充电策略核心逻辑开始 ####
    t1 = T.time_ns()

    # 获取当前车辆位置和类型
    new_location = cars[:, 1:3]
    type = cars[:, 0]

    # 初始化充电决策相关变量
    co_charge = np.zeros(number_of_cars)  # 是否进行充电的标志数组
    volt = 0  # 初始化电网电压
    short_soc_number = 0  # 电量不足的车辆计数器
    soc = np.zeros(number_of_cars)  # SOC数组
    Pv_c_max = np.zeros(number_of_cars)  # 最大充电功率数组
    charging_rate = np.zeros(number_of_cars)  # 充电率数组

    ###这里很重要，用来判断是否放电###
    # 根据车辆类型和状态计算最大充电功率，并决定是否需要充电
    for i in range(number_of_cars):
        soc[i] = cars[i, 3]  # 从车辆状态中获取当前SOC
        Pv_c_max[i] = find_Pv_c_max(eletricity_utilization_info[int(type[i]), 0], eletricity_utilization_info[int(type[i]), 1], soc[i])
        if new_location[i, 0] == 1 and new_location[i, 1] == 1:  # 如果车辆在充电站点
            co_charge[i] = 1  # 设置为充电状态

    # 如果存在需要充电的车辆
    if max(co_charge) != 0:
        # 针对额外天数和正常天数处理充电逻辑
        if day <= extra_days:
            # 检查额外天数的电源供应情况
            if extra_days_power_supply[int(day - 1), int(time / step)] > 0:
                # 计算充电所需电压，如需调整则使用二分法求解
                if volt_calculation(U_c_max, number_of_cars, extra_days_power_supply[int(day - 1), int(time / step)], soc, co_charge, Pv_c_max) < 0:
                    volt = U_c_max
                else:
                    volt = bisection_solve(volt_calculation, U_c_max, U_d_max, number_of_cars, extra_days_power_supply[int(day - 1), int(time / step)], soc, co_charge, Pv_c_max)
                # 计算并更新每辆车的充电率
                for i in range(number_of_cars):
                    charging_rate[i] = co_charge[i] * power_calculation(Pc_c_max, Pc_d_max, Pv_c_max[i], Pv_d_max, U_d_max, U_mid0, U_c_max, dU, k1, k2, volt, soc[i])
        else:
            # 正常天数的电源供应情况处理
            if max_power_supply[int(day - extra_days - 1), int(time / step)] > 0:
                # 同样计算充电所需电压
                if volt_calculation(U_c_max, number_of_cars, max_power_supply[int(day - extra_days - 1), int(time / step)], soc, co_charge, Pv_c_max) < 0:
                    volt = U_c_max
                else:
                    volt = bisection_solve(volt_calculation, U_c_max, U_d_max, number_of_cars, max_power_supply[int(day - extra_days - 1), int(time / step)], soc, co_charge, Pv_c_max)
                # 更新每辆车的充电率
                for i in range(number_of_cars):
                    charging_rate[i] = co_charge[i] * power_calculation(Pc_c_max, Pc_d_max, Pv_c_max[i], Pv_d_max, U_d_max, U_mid0, U_c_max, dU, k1, k2, volt, soc[i])

    # 更新和记录电池状态和充电统计
    for i in range(number_of_cars):
        if soc[i] < min_power_ratio:
            short_soc_number += 1

    # 记录24小时内的电压和SOC状态
    voltage_24h[int(time / step)] = volt
    min_soc_24h[int(time / step)] = min(soc)
    short_soc_24h[int(time / step)] = short_soc_number

        # 根据充电情况更新车辆状态和相关统计数据
    for i in range(number_of_cars):
        if co_charge[i] == 1:  # 车辆在充电位置
            if ((pre_location[i, 0] == 1 and pre_location[i, 1] == 1) or time - pre_arriving[i] < step):
                if pre_location[i, 0] == 1 and pre_location[i, 1] == 1:
                    if soc[i] > 0.999:
                        charge_result = soc[i]  # 避免过充
                    else:
                        charge_result = charge(charging_rate[i], step, soc[i], i)
                    if day > extra_days:
                        total_power_used[0] += charging_rate[i] * step
                    if charge_result > 1:
                        cars[i, 3] = 1  # 防止过充
                    else:
                        cars[i, 3] = charge_result
                else:
                    if soc[i] > 0.999:
                        charge_result = soc[i]
                    else:
                        charge_result = charge(charging_rate[i], (time - pre_arriving[i]), soc[i], i)
                    if day > extra_days:
                        total_power_used[0] += charging_rate[i] * (time - pre_arriving[i])
                    if charge_result > 1:
                        cars[i, 3] = 1
                    else:
                        cars[i, 3] = charge_result
                    if day > extra_days and day % 7 != 1 and day % 7 != 0:
                        length_of_stay_at_office_24h[i] += time - pre_arriving[i]
        
            # 根据SOC级别更新停车场的车辆统计
            if cars[i, 3] > 0.9:
                cars_at_office_24h[0, int(time/step)] += 1
            elif cars[i, 3] > 0.8:
                cars_at_office_24h[1, int(time/step)] += 1
            elif cars[i, 3] > 0.7:
                cars_at_office_24h[2, int(time/step)] += 1
            elif cars[i, 3] > 0.6:
                cars_at_office_24h[3, int(time/step)] += 1
            elif cars[i, 3] > 0.5:
                cars_at_office_24h[4, int(time/step)] += 1
            elif cars[i, 3] > 0.4:
                cars_at_office_24h[5, int(time/step)] += 1
            elif cars[i, 3] > 0.3:
                cars_at_office_24h[6, int(time/step)] += 1
            elif cars[i, 3] > 0.2:
                cars_at_office_24h[7, int(time/step)] += 1
            elif cars[i, 3] > 0.1:
                cars_at_office_24h[8, int(time/step)] += 1
            else:
                cars_at_office_24h[9, int(time/step)] += 1
            
            # 记录和统计充电数据
        cars_electricity24h[i, int(time / step)] = cars[i, 3]  # 更新每辆车的电量记录
        if day > extra_days:
            strategy_24h[i, int(time / step)] += charging_rate[i]  # 记录策略执行数据
            if cars[i, 1] == cars[i, 2]:
                position_24h[i, int(time / step)] = cars[i, 1]  # 记录位置信息
            else:
                position_24h[i, int(time / step)] = 3  # 表示车辆在移动中 
        
        # 特殊天数数据处理，例如第100天
        if day == 100:
            for i in range(number_of_cars):
                strategy_special[i, int(time / step)] += charging_rate[i]  # 特殊日的策略执行数据

    # 计算和记录建筑的能耗
    if day > extra_days:
        building_power_consumption_24h[int(time / step)] += np.sum(charging_rate)
        total_power_generation[0] += max_power_supply[int(day - extra_days - 1), int(time / step)] * step
        if max_power_supply[int(day - extra_days - 1), int(time / step)] > 0:
            power_consumption_rate_24h[int(time / step)] = np.sum(charging_rate) / max_power_supply[int(day - extra_days - 1), int(time / step)]

    # 总充电时间记录
    t2 = T.time_ns()
    charging_time[day-1] += round((t2-t1)/1e9, 2)
        

# %%
#计算电动车的功耗
def EV_power_consumption(V, G):
    # V: 车辆速度 (km/h)
    # G: 车辆重力 (N)
    # 计算空气阻力造成的功耗
    P_A = (0.5 * 0.35 * 1.225 * 2.097 * ((V / 3.6) ** 3)) / 1000
    # 计算传动系统损耗
    P_dr = 0.000000959656 * (V ** 3) + 0.000193052 * (V ** 2) + 0.0182062 * V + 0.375
    # 计算轮胎摩擦造成的功耗
    P_T = G * 0.0089 * (V / 3.6) / 1000
    # 计算辅助设备使用的功率
    P_anc = 3.2  # 包括空调、灯光、音响、电池管理系统等
    # 总功耗
    P = P_A + P_dr + P_T + P_anc
    return P

# %%
#模拟电动车的放电过程
def discharge(P_d, time, init_SOC, i):
    # P_d: 需要放电的功率
    # time: 放电时间 (h)
    # init_SOC: 初始状态的电池电量 (SOC)
    # i: 车辆索引
    SOC = [init_SOC]  # 初始化SOC列表
    dT = time
    step = 15  # 定义计算步长
    i_d = []  # 存储放电电流数据
    n_s_i = eletricity_utilization_info[int(cars[i, 0]), 0]
    n_p_i = eletricity_utilization_info[int(cars[i, 0]), 1]

    # 定义计算放电电流的函数
    def f1(X):
        # X为当前时刻电池的放电电流
        # 返回电池电压与负载之间的差异
        return (E0 - R * X - K / SOC[0] * (q * (1 - SOC[0]) + X) + A * np.exp(-B * q * (1 - SOC[0]))) * i_d - P_d * 1000 / (n_s_i * n_p_i)

    def f2(X):
        # 用于迭代计算每一步的放电电流
        return (E0 - R * X - K / (SOC[j - 1] - (X + i_d[j - 1]) / 2 * dT / step) * (q * (1 - (SOC[j - 1] - (X + i_d[j - 1]) / 2 * dT / step)) + X) + A * np.exp(-B * q * (1 - (SOC[j - 1] - (X + i_d[j - 1]) / 2 * dT / step)))) * X - P_d * 1000 / (n_s_i * n_p_i)

    # 使用数值方法求解初始放电电流
    i_d.append(scipy.optimize.root(f1, [0.1], method='hybr').x)

    # 对每个计算步长进行放电过程的模拟
    for j in range(1, step + 1):
        i_d.append(scipy.optimize.root(f2, [0.1], method='hybr').x)
        SOC.append(SOC[j - 1] - (i_d[j] + i_d[j - 1]) / 2 * dT / step / q)

    # 返回最终的SOC
    SOC_out = SOC[-1]
    return SOC_out

# %%
#模拟电动车的充电过程
def charge(P_c, time, init_SOC, i):
    # P_c: 充电功率
    # time: 充电时间 (h)
    # init_SOC: 初始SOC
    # i: 车辆索引
    t1 = T.time_ns()
    SOC = []
    SOC.append(init_SOC)  # 初始化SOC列表
    dT = time
    step = 15  # 定义计算步长
    i_c = []  # 存储充电电流数据
    n_s_i = eletricity_utilization_info[int(cars[i, 0]), 0]
    n_p_i = eletricity_utilization_info[int(cars[i, 0]), 1]

    # 定义计算充电电流的函数
    def f1(X):
        return (E0 - R * X - K / (0.9 - SOC[0]) * X - K / SOC[0] * q * (1 - SOC[0]) + A * np.exp(-B * q * (1 - SOC[0]))) * X - P_c * 1000 / (n_s_i * n_p_i)

    def f2(X):
        return (E0 - R * X - K / (0.9 - (SOC[j - 1] + (X + i_c[j - 1]) / 2 * dT / step)) * X - K / (SOC[j - 1] + (X + i_c[j - 1]) / 2 * dT / step) * q * (1 - (SOC[j - 1] + (X + i_c[j - 1]) / 2 * dT / step)) + A * np.exp(-B * q * (1 - (SOC[j - 1] + (X + i_c[j - 1]) / 2 * dT / step)))) * X - P_c * 1000 / (n_s_i * n_p_i)

    # 使用数值方法求解初始充电电流
    i_c.append(scipy.optimize.root(f1, [0.1], method='hybr').x)

    # 对每个计算步长进行充电过程的模拟
    for j in range(1, step + 1):
        i_c.append(scipy.optimize.root(f2, [0.1], method='hybr').x)
        SOC.append(SOC[j - 1] + (i_c[j] + i_c[j - 1]) / 2 * dT / step / q)

    # 返回最终的SOC
    SOC_out = SOC[-1]
    t2 = T.time_ns()
    charge_time[day-1] += (t2-t1)/1e9
    charge_num[day-1] += 1
    return SOC_out

# %%
#计算电动车的最大充电功率
def find_Pv_c_max(n_s, n_p, SOC):
    # n_s: 串联电池数
    # n_p: 并联电池数
    # SOC: 电池状态的电量
    Pv_c_max0 = 40  # 基准最大充电功率 (kW)
    if SOC == 1:
        Pv_c_max = 0
    elif SOC >= 0.95:
        U = (E0 - R * 0.01 * q + A) * n_s
        I_max = Pv_c_max0 * 1000 / U
        I_min = 0.01 * q * n_p
        Pv_c_max = ((SOC - 0.95) / 0.05 * (I_min - I_max) + I_max) * U / 1000
    else:
        Pv_c_max = Pv_c_max0

    return Pv_c_max


# %%
#根据车辆运行状态计算其能耗
def power_consumption(day, time, i, time1, time2):
    # day: 当前天数
    # time: 当前时间
    # i: 车辆索引
    # time1: 开始时间
    # time2: 结束时间
    type = cars[i, 0]
    init_soc = cars[i, 3]
    # 根据时间选择适当的速度数据
    if time1 < 0:
        speed = yesterday_speed[int((time1+24)/step), i]
    else:
        speed = today_speed[int(time1/step), i]
    discharge_rate = EV_power_consumption(speed, eletricity_utilization_info[int(type)][2])
    # 根据放电率更新车辆的SOC
    if time2 - time1 < step:
        cars[i, 3] = discharge(discharge_rate, (time2 - time1), init_soc, i)
    else:
        cars[i, 3] = discharge(discharge_rate, step, init_soc, i)
    # 检查电量状态并记录
    if cars[i, 3] < min_power_ratio:
        print(f"Error! Driver {i}'s car does not have enough electricity on day {day - extra_days} at {time}!")
    if cars[i, 3] <= 0:
        print(f"Error! Driver {i}'s car is broken down on day {day - extra_days} at {time}!")
    if day > extra_days:
        discharge_strategy_24h[i, int(time/step)] = discharge_rate


# %% [markdown]
# ### 各种车行为相关函数

# %%
#根据时间和工作日/周末选择合适的马尔可夫转移矩阵
def pick_Markov_matrix(i, day, time, to_office, at_office):
    # i: 车辆索引
    # day: 当前天数
    # time: 当前时间
    # to_office: 去办公室的时间
    # at_office: 在办公室的时间

    # 判断是周末还是工作日，并选择相应的马尔可夫矩阵
    if day % 7 == 1 or day % 7 == 0:
        if Alpha_go_out_leisure_time[i] == 0:
            Markov_matrix_list = Markov_matirx_list_weekend_negative
        else:
            Markov_matrix_list = Markov_matirx_list_weekend_positive
    else:
        if Alpha_go_out_during_work_time[i] == 1:
            Markov_matrix_list = Markov_matirx_list_weekday_negative
        else:
            Markov_matrix_list = Markov_matirx_list_weekday_positive

    # 根据当前时间选择适用的转移矩阵
    if time < to_office:
        return Markov_matrix_list[0]
    elif (time < to_office + step or time < at_office) and to_office < 12:
        return Markov_matrix_list[1]
    elif time < 12:
        return Markov_matrix_list[2]
    elif time < 13:
        return Markov_matrix_list[3]
    elif time < 18:
        return Markov_matrix_list[4]
    elif time < 21:
        if Alpha_go_out_leisure_time[i] == 2:
            return Markov_matrix_list[5]
        elif Alpha_go_out_leisure_time[i] == 0:
            return Markov_matrix_list[6]
        else:
            return Markov_matrix_list[7]
    else:
        if night_before_other[i] == 0:
            return Markov_matrix_list[8]
        else:
            return Markov_matrix_list[9]


# %%
#根据状态概率决定车辆的下一个位置
def determine_next_location(status):
    # status: 当前状态的概率分布
    n = np.random.rand()  # 生成随机数以模拟状态转移
    if n < status[0]:
        return 0  # 表示车辆在家
    elif n < status[0] + status[1]:
        return 1  # 表示车辆在办公室
    else:
        return 2  # 表示车辆在路上


# %%
#生成车辆到达办公室的时间分布
def generate_the_arriving_time(number_of_cars):
    # number_of_cars: 车辆数量
    # 返回正态分布生成的到达时间，平均时间为8.5小时，标准差为0.4小时
    return np.random.normal(8.5, 0.4, number_of_cars)


# %%
# 用于生成从家或办公室到其他地方的距离，根据是否为周末以及当前时间使用不同的分布参数来模拟距离
def generate_otherplace_distance(time, day, number_of_cars):
    # 根据周末或工作日使用不同的分布参数来生成距离
    if day % 7 == 1 or day % 7 == 0:  # 周末
        if time < 8:
            home_to_other = np.random.gamma(1.14, 8.58974, number_of_cars)*k_weekend_h_o
        elif time < 12:
            home_to_other = np.random.gamma(1.14, 8.58974, number_of_cars)*k_weekend_h_o
        elif time < 13:
            home_to_other = np.random.gamma(1.14, 8.58974, number_of_cars)*k_weekend_h_o
        elif time < 18:
            home_to_other = np.random.gamma(1.14, 8.58974, number_of_cars)*k_weekend_h_o
        elif time < 21:
            home_to_other = np.random.gamma(1.14, 8.58974, number_of_cars)*k_weekend_h_o
        else:
            home_to_other = np.random.gamma(1.14, 8.58974, number_of_cars)*k_weekend_h_o
        # 根据不同的时间段选择不同的分布参数来模拟从办公室到其他地方的距离
        if time < 8:
            office_to_other = np.random.gamma(1.8, 2.86666, number_of_cars)
        elif time < 12:
            office_to_other = np.random.gamma(1.8, 2.86666, number_of_cars)
        elif time < 13:
            office_to_other = np.random.gamma(1.9, 1.579, number_of_cars)
        elif time < 18:
            office_to_other = np.random.gamma(1.8, 2.86666, number_of_cars)
        elif time < 21:
            office_to_other = np.random.gamma(1.35, 2.40180, number_of_cars)
        else:
            office_to_other = np.random.gamma(1.35, 2.40180, number_of_cars)
    else:  # 工作日
        if time < 8:
            home_to_other = np.random.gamma(1.3, 6.44570, number_of_cars)
        elif time < 12:
            home_to_other = np.random.gamma(1.3, 6.44570, number_of_cars)
        elif time < 13:
            home_to_other = np.random.gamma(1.3, 6.44570, number_of_cars)
        elif time < 18:
            home_to_other = np.random.gamma(1.3, 6.44570, number_of_cars)
        elif time < 21:
            home_to_other = np.random.gamma(1.3, 6.44570, number_of_cars)*k_after_work_h_o #表示在下班后从家去其它地点这个行为比较特殊
        else:
            home_to_other = np.random.gamma(1.3, 6.44570, number_of_cars)
        # 根据不同的时间段选择不同的分布参数来模拟从办公室到其他地方的距离
        if time < 8:
            office_to_other = np.random.gamma(1.8, 2.86666, number_of_cars)
        elif time < 12:
            office_to_other = np.random.gamma(1.8, 2.86666, number_of_cars) * Alpha_go_out_during_work_time
        elif time < 13:
            office_to_other = np.random.gamma(1.9, 1.579, number_of_cars)
        elif time < 18:
            office_to_other = np.random.gamma(1.8, 2.86666, number_of_cars) * Alpha_go_out_during_work_time
        elif time < 21:
            office_to_other = np.random.gamma(1.35, 2.40180, number_of_cars) * k_after_work_w_o
        else:
            office_to_other = np.random.gamma(1.35, 2.40180, number_of_cars)

    # 确保生成的距离不为负
    home_to_other[home_to_other < 0] = 0
    office_to_other[office_to_other < 0] = 0

    # 返回从家到其他地方和从办公室到其他地方的距离
    return home_to_other, office_to_other


# %%
#计算从一个地点到另一个地点所需的时间
def find_time_spent(now, next, time, i):
    # 初始化花费的时间和距离
    time_spent = 0
    n = 0  # 时间步长的索引
    speed = today_speed  # 当天的速度数据

    # 根据当前位置和下一位置决定行驶的距离
    if (now == 0 and next == 1) or (now == 1 and next == 0):
        distance = home_to_office_distance[int(i)]
    elif (now == 0 and next == 2) or (now == 2 and next == 0):
        distance = home_to_otherplace_distance[int(i)]
    else:
        distance = office_to_otherplace_distance[int(i)]

    # 计算行驶时间，根据车速减少距离，增加时间
    while distance > 0:
        current_speed = speed[int(time / step + n), i]
        if distance - current_speed * step > 0:
            time_spent += step
        else:
            time_spent += distance / current_speed
        distance -= current_speed * step
        n += 1
        # 如果超过一天，调整索引和速度数据
        if int(time / step + n) >= int(24/step):
            n -= int(24/step)
            speed = tomorrow_speed

    return time_spent


# %%
#计算车辆离开办公室回家的时间
def find_leaving_time(at_office_time, home_to_office_distance, today_speed):
    leaving_time = np.copy(at_office_time)
    for i in range(number_of_cars):
        distance = home_to_office_distance[i]
        time_stamp = at_office_time[i]
        n = 0
        while distance > 0:
            # 计算剩余距离和相应的离开时间
            if n == 0:  # 首次计算速度和时间
                speed = today_speed[int(time_stamp/step), i]
                n += 1
                if distance - (time_stamp/step - int(time_stamp/step)) * step * speed <= 0:
                    leaving_time[i] -= distance / speed
                    break
                distance -= (time_stamp / step - int(time_stamp / step)) * step * speed
                leaving_time[i] -= (time_stamp / step - int(time_stamp / step)) * step
            else:  # 往后每个时间步长的处理
                speed = today_speed[int(time_stamp/step - n), i]
                n += 1
                if distance - step * speed <= 0:
                    leaving_time[i] -= distance / speed
                    break
                distance -= step * speed
                leaving_time[i] -= step
    return leaving_time


# %%
#根据当前和下一个位置更新行驶距离和统计数据。
def update_distance(now, next, time, i):
    # 根据是工作日还是周末来更新行驶距离
    if day % 7 != 1 and day % 7 != 0:  # 工作日
        # 更新不同时间段的行驶距离
        if time < 12:
            # 从家到办公室或从办公室到家的距离统计
            if (next == 0 and now == 1) or (next == 1 and now == 0):
                traveling_home_to_office_weekday[0, i] += home_to_office_distance[i]
                traveling_distance_weekday[0, i] += home_to_office_distance[i]
                commuting[i, day-extra_days-1] += 1
            # 从办公室到其他地方的距离统计
            elif next == 2 and now == 1:
                morning[0] += 1
                traveling_distance_weekday[0, i] += 2*office_to_otherplace_distance[i]
                traveling_distance_weekday[3, i] += 2*office_to_otherplace_distance[i]
        # 中午时间段的距离更新
        elif 12 <= time < 13:
            if (next == 0 and now == 2) or (next == 2 and now == 0):
                traveling_distance_weekday[0, i] += home_to_otherplace_distance[i]
            elif next == 2 and now == 1:
                noon[0] += 1
                traveling_distance_weekday[0, i] += 2*office_to_otherplace_distance[i]
                traveling_distance_weekday[7, i] += 2*office_to_otherplace_distance[i]
        # 下午时间段的距离更新
        elif 13 <= time < 18:
            if next == 2 and now == 1:
                afternoon_out[0] += 1
                traveling_distance_weekday[0, i] += 2*office_to_otherplace_distance[i]
                traveling_distance_weekday[4, i] += 2*office_to_otherplace_distance[i]
        # 晚上时间段的距离更新
        else:
            if (next == 0 and now == 1) or (next == 1 and now == 0):
                traveling_home_to_office_weekday[0, i] += home_to_office_distance[i]
                traveling_distance_weekday[0, i] += home_to_office_distance[i]
                commuting[i, day-extra_days-1] += 1
            elif next == 2 and now == 0:
                evening1[0] += 1
                traveling_distance_weekday[0, i] += 2*home_to_otherplace_distance[i]
                traveling_distance_weekday[6, i] += 2*home_to_otherplace_distance[i]
            elif next == 2 and now == 1:
                evening2[0] += 1
                traveling_distance_weekday[0, i] += 2*office_to_otherplace_distance[i]
                traveling_distance_weekday[5, i] += 2*office_to_otherplace_distance[i]
    else:  # 周末
        # 更新从家到办公室或办公室到家的距离
        if (next == 0 and now == 1) or (next == 1 and now == 0):
            if time < 12 or 13 <= time < 18:
                traveling_home_to_office_weekend[0, i] += home_to_office_distance[i]
                traveling_distance_weekend[0, i] += home_to_office_distance[i]
            elif 12 <= time < 13:
                traveling_home_to_office_weekend[1, i] += home_to_office_distance[i]
                traveling_distance_weekend[1, i] += home_to_office_distance[i]
            else:
                traveling_home_to_office_weekend[2, i] += home_to_office_distance[i]
                traveling_distance_weekend[2, i] += home_to_office_distance[i]
        # 更新从家或办公室到其他地方的距离
        elif (next == 0 and now == 2) or (next == 2 and now == 0):
            if time < 12 or 13 <= time < 18:
                traveling_distance_weekend[0, i] += home_to_otherplace_distance[i]
            elif 12 <= time < 13:
                traveling_distance_weekend[1, i] += home_to_otherplace_distance[i]
            else:
                traveling_distance_weekend[2, i] += home_to_otherplace_distance[i]
        else:
            if time < 12 or 13 <= time < 18:
                traveling_distance_weekend[0, i] += office_to_otherplace_distance[i]
            elif 12 <= time < 13:
                traveling_distance_weekend[1, i] += office_to_otherplace_distance[i]
            else:
                traveling_distance_weekend[2, i] += office_to_otherplace_distance[i]


# %%
#记录司机是否已经到达办公室
def update_have_been_to_office(now, i):
    if now == 1:  # 如果当前位置是办公室，则更新状态
        have_been_to_the_office[i] = 1


# %%
#根据当前时间和到达时间来更新司机在不同地点的停留时间
def record_time(time, arriving_time, i):
    # 根据司机的当前位置和计划到达的时间更新停留时间
    if time >= arriving_time:
        if cars[i, 1] == cars[i, 2] == 0:
            home_time_24h[i] += step
        elif cars[i, 1] == cars[i, 2] == 1:
            office_time_24h[i] += step
        elif cars[i, 1] == cars[i, 2] == 2:
            other_time_24h[i] += step
    # 如果即将到达，更新在途时间和目的地停留时间
    elif arriving_time - time < step:
        on_the_way_time_24h[i] += arriving_time - time
        if destination[i] == 0:
            home_time_24h[i] += step - (arriving_time - time)
        elif destination[i] == 1:
            office_time_24h[i] += step - (arriving_time - time)
        elif destination[i] == 2:
            other_time_24h[i] += step - (arriving_time - time)
    # 如果完全在途中，则更新在途时间
    elif arriving_time - time >= step:
        on_the_way_time_24h[i] += step


# %%
#调整司机离开家的时间，考虑到一部分人夜间不会外出
def adjust_leaving_time(leaving_time):
    unmoved_rate = 0.24  # 未移动的司机比率
    index = np.arange(0, number_of_cars, 1).tolist()  # 创建司机索引列表
    night = 0  # 夜间在家的司机数
    at_home = []  # 在家的司机列表

    # 移除夜间仍在办公室的司机
    for i in range(number_of_cars):
        if cars[i, 2] == 1:
            index.remove(i)
            night += 1

    # 根据未移动比率随机选择司机晚归
    if int(unmoved_rate*number_of_cars)-night > 0:
        for i in range(int(unmoved_rate*number_of_cars)-night):
            x = np.random.randint(0, len(index))
            at_home.append(index[x])
            index.remove(index[x])

    # 调整被选择司机的离开时间
    for i in at_home:
        leaving_time[i] = 18  # 设置晚归时间

    if day % 7 != 1 and day % 7 != 0:
        night_record.append(night)

    print(int(unmoved_rate*number_of_cars)-night)
    print(leaving_time)
    
    return leaving_time


# %%
#分配上班期间积极外出的司机
def go_out_during_work_time(percentage_to_go_out_during_working, number_of_cars):
    go_out_during_working = []
    index = np.arange(0, number_of_cars, 1).tolist()

    # 根据比例随机选择上班时间外出的司机
    for i in range(int(percentage_to_go_out_during_working*number_of_cars)):
        x = np.random.randint(0, len(index))
        go_out_during_working.append(index[x])
        index.remove(index[x])

    # 初始化结果数组，并为选定的司机分配外出标记
    result = np.ones(number_of_cars)
    for j in go_out_during_working:
        result[j] = np.random.gamma(1.1, 0.909091)  # 使用Gamma分布赋予外出倾向

    return result


# %%
#分配从家去其他地方的积极外出司机
def go_out_leisure_time(percentage_to_go_out_leisure_time, number_of_cars):
    go_out_leisure = []
    index = np.arange(0, number_of_cars, 1).tolist()

    # 随机选择休闲时间外出的司机
    for i in range(int(percentage_to_go_out_leisure_time*number_of_cars)):
        x = np.random.randint(0, len(index))
        go_out_leisure.append(index[x])
        index.remove(index[x])

    # 调用函数进一步处理这些司机的外出倾向
    go_out_leisure_P1 = go_out_leisure_time_P1(percentage_to_go_out_leisure_time_P1, go_out_leisure)

    # 初始化结果数组，并为选定的司机分配外出倾向级别
    result = np.zeros(number_of_cars)
    for j in go_out_leisure:
        result[j] = 1
    for k in go_out_leisure_P1:
        result[k] = 2

    return result


# %%
#进一步细化下班后最积极外出的司机群体
def go_out_leisure_time_P1(percentage_to_go_out_leisure_time_P1, go_out_leisure):
    go_out_leisure_P1 = []

    # 从已选的休闲时间外出司机中进一步选择
    for i in range(int(percentage_to_go_out_leisure_time_P1*len(go_out_leisure))):
        x = np.random.randint(0, len(go_out_leisure))
        go_out_leisure_P1.append(go_out_leisure[x])
        go_out_leisure.remove(go_out_leisure[x])

    return go_out_leisure_P1


# %%
# 初始化上班期间和休闲时间的外出行为模式
Alpha_go_out_during_work_time = go_out_during_work_time(percentage_to_go_out_during_working, number_of_cars)
Alpha_go_out_leisure_time = go_out_leisure_time(percentage_to_go_out_leisure_time, number_of_cars)

# 打印验证数据
print(Alpha_go_out_during_work_time)
print(Alpha_go_out_leisure_time)

# %% [markdown]
# #### 马尔可夫矩阵链

# %%
#Intialize the state representation of each place
location_status = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

Markov_matirx_list_weekday_positive = []

#Martix0 0:00 - leaving time
Markov_matirx_list_weekday_positive.append(np.array([[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0, 1]]))

#Martix1 leaving time - arriving time (to office)
Markov_matirx_list_weekday_positive.append(np.array([[0, 1, 0],
                                                     [0, 1, 0],
                                                     [0, 1, 0]]))

#Martix2 arriving time - 12:00
Markov_matirx_list_weekday_positive.append(np.array([[1, 0, 0],
                                                     [0, 0.975, 0.025],
                                                     [0, 0.4, 0.6]]))

# 在办公室0.914
#Martix3 12:00 - 13:00
Markov_matirx_list_weekday_positive.append(np.array([[1, 0, 0],
                                                     [0, 0.985, 0.015],
                                                     [0, 0.02, 0.98]]))

#Martix4 13:00 - 18:00
Markov_matirx_list_weekday_positive.append(np.array([[1, 0, 0],
                                                     [0, 0.982, 0.018],
                                                     [0, 0.4, 0.6]]))

#下班后即可能从家也可能从办公室出去其它0.85
#Martix5 18:00 - 21:00
Markov_matirx_list_weekday_positive.append(np.array([[0.893, 0, 0.107],
                                                     [0.082, 0.863, 0.055],
                                                     [0.01, 0, 0.99]]))

#下班后不去其它0.05
#Martix6 18:00 - 21:00
Markov_matirx_list_weekday_positive.append(np.array([[1, 0, 0],
                                                     [0.1, 0.9, 0],
                                                     [0, 0, 1]]))

#下班后只从家出去0.10
#Martix7 18:00 - 21:00
Markov_matirx_list_weekday_positive.append(np.array([[0.9, 0, 0.1],
                                                     [0.1, 0.9, 0],
                                                     [0, 0, 1]]))

# 0.85
# 从家去的其他
#Martix8 21:00 - 24:00
Markov_matirx_list_weekday_positive.append(np.array([[1, 0, 0],
                                                     [0.01, 0.99, 0],
                                                     [0.5, 0, 0.5]]))

# 从办公室去的其他
#Martix9 21:00 - 24:00
Markov_matirx_list_weekday_positive.append(np.array([[1, 0, 0],
                                                     [0.01, 0.99, 0],
                                                     [0.5, 0, 0.5]]))

Markov_matirx_list_weekday_negative = []

#Martix0 0:00 - leaving time
Markov_matirx_list_weekday_negative.append(np.array([[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0, 1]]))

#Martix1 leaving time - arriving time (to office)
Markov_matirx_list_weekday_negative.append(np.array([[0, 1, 0],
                                                     [0, 1, 0],
                                                     [0, 1, 0]]))

#Martix2 arriving time - 12:00
Markov_matirx_list_weekday_negative.append(np.array([[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0, 1]]))

# 在办公室0.914
#Martix3 12:00 - 13:00
Markov_matirx_list_weekday_negative.append(np.array([[1, 0, 0],
                                                     [0, 0.985, 0.015],
                                                     [0, 0.02, 0.98]]))

#Martix4 13:00 - 18:00
Markov_matirx_list_weekday_negative.append(np.array([[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0.5, 0.5]]))

#下班后即可能从家也可能从办公室出去其它0.85
#Martix5 18:00 - 21:00
Markov_matirx_list_weekday_negative.append(np.array([[0.893, 0, 0.107],
                                                     [0.082, 0.863, 0.055],
                                                     [0.01, 0, 0.99]]))

#下班后不去其它0.05
#Martix6 18:00 - 21:00
Markov_matirx_list_weekday_negative.append(np.array([[1, 0, 0],
                                                     [0.1, 0.9, 0],
                                                     [0, 0, 1]]))

#下班后只从家出去0.10
#Martix7 18:00 - 21:00
Markov_matirx_list_weekday_negative.append(np.array([[0.9, 0, 0.1],
                                                     [0.1, 0.9, 0],
                                                     [0, 0, 1]]))

# 0.85
# 从家去的其他
#Martix8 21:00 - 24:00
Markov_matirx_list_weekday_negative.append(np.array([[1, 0, 0],
                                                     [0.01, 0.99, 0],
                                                     [0.5, 0, 0.5]]))

# 从办公室去的其他
#Martix9 21:00 - 24:00
Markov_matirx_list_weekday_negative.append(np.array([[1, 0, 0],
                                                     [0.01, 0.99, 0],
                                                     [0.5, 0, 0.5]]))

Markov_matirx_list_weekend_positive = []

#Martix0 0:00 - leaving
Markov_matirx_list_weekend_positive.append(np.array([[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0, 1]]))

#Martix1 leaving time - arriving time (to office)
Markov_matirx_list_weekend_positive.append(np.array([[0.996, 0.004, 0],
                                                     [0, 0.98, 0.02],
                                                     [0.07, 0, 0.93]]))

#Martix2 arriving - 12:00
Markov_matirx_list_weekend_positive.append(np.array([[0.985, 0.005, 0.01],
                                                     [0, 0.97, 0.03],
                                                     [0.01, 0.01, 0.98]]))

#Martix3 12:00 - 13:00
Markov_matirx_list_weekend_positive.append(np.array([[1, 0, 0],
                                                     [0, 0.985, 0.015],
                                                     [0, 0.02, 0.98]]))

#Martix4 13:00 - 18:00
Markov_matirx_list_weekend_positive.append(np.array([[0.98, 0, 0.02],
                                                     [0, 0.99, 0.01],
                                                     [0.01, 0.01, 0.98]]))

#下班后即可能从家也可能从办公室出去其它0.85
#Martix5 18:00 - 21:00
Markov_matirx_list_weekend_positive.append(np.array([[0.999, 0, 0.001],
                                                     [0.02, 0.979, 0.001],
                                                     [0.01, 0, 0.99]]))
#下班后不去其它0.05
#Martix6 18:00 - 21:00
Markov_matirx_list_weekend_positive.append(np.array([[1, 0, 0],
                                                     [0.05, 0.95, 0],
                                                     [0, 0, 1]]))

#下班后只从家出去0.10
#Martix7 18:00 - 21:00
Markov_matirx_list_weekend_positive.append(np.array([[0.999, 0, 0.001],
                                                     [0.05, 0.95, 0],
                                                     [0, 0, 1]]))

#Martix8 21:00 - 24:00
Markov_matirx_list_weekend_positive.append(np.array([[1, 0, 0],
                                                     [0.05, 0.95, 0],
                                                     [0.49, 0.01, 0.5]]))

#Martix9 21:00 - 24:00
Markov_matirx_list_weekend_positive.append(np.array([[1, 0, 0],
                                                     [0.05, 0.95, 0],
                                                     [0.49, 0.01, 0.5]]))

Markov_matirx_list_weekend_negative = []

#Martix0 0:00 - leaving
Markov_matirx_list_weekend_negative.append(np.array([[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0, 1]]))

#Martix1 leaving time - arriving time (to office)
Markov_matirx_list_weekend_negative.append(np.array([[0.996, 0.004, 0],
                                                     [0, 0.98, 0.02],
                                                     [0.07, 0, 0.93]]))

#Martix2 arriving - 12:00
Markov_matirx_list_weekend_negative.append(np.array([[0.996, 0.004, 0],
                                                     [0, 1, 0],
                                                     [0, 0, 1]]))

#Martix3 12:00 - 13:00
Markov_matirx_list_weekend_negative.append(np.array([[1, 0, 0],
                                                     [0, 0.985, 0.015],
                                                     [0, 0.02, 0.98]]))

#Martix4 13:00 - 18:00
Markov_matirx_list_weekend_negative.append(np.array([[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0.05, 0.95]]))

#下班后即可能从家也可能从办公室出去其它0.85
#Martix5 18:00 - 21:00
Markov_matirx_list_weekend_negative.append(np.array([[0.999, 0, 0.001],
                                                     [0.02, 0.979, 0.001],
                                                     [0.01, 0, 0.99]]))

#下班后不去其它0.05
#Martix6 18:00 - 21:00
Markov_matirx_list_weekend_negative.append(np.array([[1, 0, 0],
                                                     [0.05, 0.95, 0],
                                                     [0, 0, 1]]))

#下班后只从家出去0.10
#Martix7 18:00 - 21:00
Markov_matirx_list_weekend_negative.append(np.array([[0.999, 0, 0.001],
                                                     [0.05, 0.95, 0],
                                                     [0, 0, 1]]))

#Martix8 21:00 - 24:00
Markov_matirx_list_weekend_negative.append(np.array([[1, 0, 0],
                                                     [0.05, 0.95, 0],
                                                     [0.49, 0.01, 0.5]]))

#Martix9 21:00 - 24:00
Markov_matirx_list_weekend_negative.append(np.array([[1, 0, 0],
                                                     [0.05, 0.95, 0],
                                                     [0.49, 0.01, 0.5]]))


# %% [markdown]
# ### 统计和导出导入数据

# %% [markdown]
# ### 主程序，对每一天进行模拟

# %%
# 对每一天进行模拟
for day in np.arange(1, num_of_days + extra_days + 1, 1):
    T1 = T.time_ns()  # 记录开始时间
    # 初始化各种状态记录数组
    cars_at_office_24h = np.zeros((10, int(24 / step)))
    cars_electricity24h = np.zeros((number_of_cars, int(24 / step)))
    strategy_24h = np.zeros((number_of_cars, int(24 / step)))
    discharge_strategy_24h = np.zeros((number_of_cars, int(24 / step)))
    length_of_stay_at_office_24h = np.zeros(number_of_cars)
    building_power_consumption_24h = np.zeros(int(24 / step))
    position_24h = np.zeros((number_of_cars, int(24 / step)))
    home_time_24h = np.zeros(number_of_cars)
    office_time_24h = np.zeros(number_of_cars)
    other_time_24h = np.zeros(number_of_cars)
    on_the_way_time_24h = np.zeros(number_of_cars)
    voltage_24h = np.zeros(int(24 / step))
    min_soc_24h = np.zeros(int(24 / step))
    short_soc_24h = np.zeros(int(24 / step))
    power_consumption_rate_24h = np.zeros(int(24 / step))

    # 更新速度数据
    yesterday_speed = today_speed
    today_speed = tomorrow_speed
    tomorrow_speed = generate_speed_distribution(day+1, number_of_cars)

    # 生成每辆车到达办公室的时间和离开时间
    at_office_time = generate_the_arriving_time(number_of_cars)
    to_office_time = adjust_leaving_time(find_leaving_time(at_office_time, home_to_office_distance, today_speed))

    # 初始化当天每辆车是否到达过办公室的记录
    have_been_to_the_office = np.zeros(number_of_cars)
    night_before_other = np.zeros(number_of_cars)
    afternoon = np.zeros(number_of_cars)

    # 按时间步长迭代模拟车辆行为
    for time in np.arange(0, 24, step):
        # 跟踪车辆位置和电量状态
        print(time, cars[0, 3], [cars[0, 1], cars[0, 2]])
        home_to_otherplace_distance, office_to_otherplace_distance = generate_otherplace_distance(time, day, number_of_cars)
        pre_location = np.copy(cars[:, 1:3])
        pre_arriving = np.copy(arriving_time)

        # 更新车辆状态
        for i in range(number_of_cars):
            if time >= leaving_time[i]:
                cars[i, 2] = destination[i]
            if leaving_time[i] <= time < arriving_time[i]:
                if time - leaving_time[i] > step:
                    power_consumption(day, time, i, time - step, time)
                else:
                    power_consumption(day, time, i, leaving_time[i], time)
            elif time >= arriving_time[i]:
                if time - arriving_time[i] < step:
                    cars[i, 1] = destination[i]
                    power_consumption(day, time, i, time - step, arriving_time[i])
                Markov_matrix = pick_Markov_matrix(i, day, time, to_office_time[i], at_office_time[i])
                destination[i] = determine_next_location(np.matmul(location_status[int(cars[i, 1])], Markov_matrix))
                if destination[i] != cars[i, 1]:
                    leaving_time[i] = time
                    arriving_time[i] = leaving_time[i] + find_time_spent(cars[i, 1], destination[i], time, i)
                    if destination[i] == 2:
                        night_before_other[i] = np.copy(cars[i, 1])
                    if day > extra_days:
                        update_distance(cars[i, 1], destination[i], time, i)
            update_have_been_to_office(cars[i, 1], i)
            record_time(time, arriving_time[i], i)
        
        # 执行充电策略
        charging_strategy(pre_location, pre_arriving, day, time)

    # 更新每辆车的时间记录
    arriving_time -= 24
    leaving_time -= 24

    # 下午出行次数
    for i in range(number_of_cars):
        if afternoon[i] > 1:
            afternoon_times[0] += (afternoon[i] - 1)

    # 周末自给自足
    for i in range(number_of_cars):
        if have_been_to_the_office[i] == 0:
            cars[i, 3] = electricity_record[i]
            for t in range(int(24/step)):
                if cars_electricity24h[i, t] < min_power_ratio:
                    short_soc_24h[t] -= 1
                cars_electricity24h[i, t] = electricity_record[i]
        else:
            electricity_record[i] = cars[i, 3]  

    # 模拟结束后处理每辆车的数据记录
    if day > extra_days:
        # 保存每天的详细数据
        building_power_consumption.append(building_power_consumption_24h)
        power_consumption_rate.append(power_consumption_rate_24h)
        cars_at_office.append(cars_at_office_24h)
        cars_electricity.append(cars_electricity24h)
        strategy.append(strategy_24h)
        discharge_strategy.append(discharge_strategy_24h)
        length_of_stay_at_office.append(length_of_stay_at_office_24h)
        position.append(position_24h.transpose())
        home_time.append(home_time_24h)
        office_time.append(office_time_24h)
        other_time.append(other_time_24h)
        on_the_way_time.append(on_the_way_time_24h)
        voltage.append(voltage_24h)
        min_soc.append(min_soc_24h)
        short_soc.append(short_soc_24h)
        # 更多状态记录
    else:
        first_month_cars_at_office.append(cars_at_office_24h)
        first_month_cars_electricity.append(cars_electricity24h)
    # 记录执行时间
    T2 = T.time_ns()
    total_time[day-1] = round((T2-T1)/1e9, 2)
    print(day, "Total: ", total_time[day-1], " Charging: ", charging_time[day-1], " Charge: ", charge_time[day-1], " Volt: ", volt_time[day-1], " Charge_num: ", charge_num[day-1])

print("The total power generated is: {} kWh".format(total_power_generation[0]))
print("The total power used is: {} kWh".format(total_power_used[0]))
print("The used over generated ratio is: {}".format(total_power_used[0]/total_power_generation[0]))
print("Night cars: {}".format(np.linalg.norm(night_record, 1)/len(night_record)))
print("Morning out: {}".format(morning[0]))
print("Noon out: {}".format(noon[0]))
print("Afternoon out: {}".format(afternoon_out[0]))
print("Evening 1 out: {}".format(evening1[0]))
print("Evening 2 out: {}".format(evening2[0]))

# %% [markdown]
# ### 输出各类需要的分析文件

# %%
#输出各类文件
# 光伏消纳情况：计算并保存总发电量、总用电量和消纳比例
power_ratio = [total_power_generation[0], total_power_used[0], total_power_used[0]/total_power_generation[0]]
wb = op.load_workbook("D:/CODE/Liangbo/pk140_10_0_0/Power_ratio.xlsx")
sht = wb.active
sht.append(power_ratio)
wb.save("D:/CODE/Liangbo/pk140_10_0_0/Power_ratio.xlsx")

# 保存每辆车每天的电量情况到CSV，然后转存到Excel并删除原CSV文件
file = open("output/cars_electricity24h.csv", 'w')
for a in cars_electricity:
    for i in range(len(a)):
        for x in a[i]:
            file.write(f"{x},")
        file.write(f"\n")
file.close()
(pd.read_csv("output/cars_electricity24h.csv", header = None, index_col = False)).to_excel("output/cars_electricity24h.xlsx", header = None, index = False)
os.remove("output/cars_electricity24h.csv")

# 保存每辆车每天的充电策略到CSV，然后转存到Excel并删除原CSV文件
file = open("output/strategy_24h.csv", 'w')
for a in strategy:
    for i in range(len(a)):
        for x in a[i]:
            file.write(f"{x},")
        file.write(f"\n")
file.close()
(pd.read_csv("output/strategy_24h.csv", header = None, index_col = False)).to_excel("output/strategy_24h.xlsx", header = None, index = False)
os.remove("output/strategy_24h.csv")

# 保存每辆车每天的放电策略到CSV，然后转存到Excel并删除原CSV文件
file = open("output/discharge_strategy_24h.csv", 'w')
for a in discharge_strategy:
    for i in range(len(a)):
        for x in a[i]:
            file.write(f"{x},")
        file.write(f"\n")
file.close()
(pd.read_csv("output/discharge_strategy_24h.csv", header = None, index_col = False)).to_excel("output/discharge_strategy_24h.xlsx", header = None, index = False)
os.remove("output/discharge_strategy_24h.csv")

# 保存办公室每个时间段的车辆数到CSV，然后转存到Excel并删除原CSV文件
file = open("output/cars_at_office_24h.csv", 'w')
for x in cars_at_office:
    for i in range(10):
        for j in range(int(24/step)):
            file.write(rf"{x[i, j]},")
        file.write(f"\n")
file.close()
(pd.read_csv("output/cars_at_office_24h.csv", header = None, index_col = False)).to_excel("output/cars_at_office_24h.xlsx", header = None, index = False)
os.remove("output/cars_at_office_24h.csv")

# 保存建筑每个时间段的功耗到CSV，然后转存到Excel并删除原CSV文件
file = open("output/building_power_consumption_24h.csv", 'w')
for x in building_power_consumption:
    for a in x:
        file.write(f"{a},")
    file.write(f"\n")
file.close()
(pd.read_csv("output/building_power_consumption_24h.csv", header = None, index_col = False)).to_excel("output/building_power_consumption_24h.xlsx", header = None, index = False)
os.remove("output/building_power_consumption_24h.csv")

# 保存每辆车每个时间段的位置到CSV，然后转存到Excel并删除原CSV文件
file = open("output/position_24h.csv", 'w')
for a in position:
    for i in range(len(a)):
        for x in a[i]:
            file.write(f"{x},")
        file.write(f"\n")
file.close()
(pd.read_csv("output/position_24h.csv", header = None, index_col = False)).to_excel("output/position_24h.xlsx", header = None, index = False)
os.remove("output/position_24h.csv")

# 保存特殊策略信息到CSV，然后转存到Excel并删除原CSV文件
file = open("output/strategy_special.csv", 'w')
for i in range(len(strategy_special)):
    for x in strategy_special[i]:
        file.write(f"{x},")
    file.write(f"\n")
file.close()
(pd.read_csv("output/strategy_special.csv", header = None, index_col = False)).to_excel("output/strategy_special.xlsx", header = None, index = False)
os.remove("output/strategy_special.csv")

# 保存电压数据到CSV，然后转存到Excel并删除原CSV文件
file = open("output/voltage.csv", 'w')
for x in voltage:
    for a in x:
        file.write(f"{a},")
    file.write(f"\n")
file.close()
(pd.read_csv("output/voltage.csv", header = None, index_col = False)).to_excel("output/voltage.xlsx", header = None, index = False)
os.remove("output/voltage.csv")

# 保存功率消耗率数据到CSV，然后转存到Excel并删除原CSV文件
file = open("output/power_consumption_rate.csv", 'w')
for x in power_consumption_rate:
    for a in x:
        file.write(f"{a},")
    file.write(f"\n")
file.close()
(pd.read_csv("output/power_consumption_rate.csv", header = None, index_col = False)).to_excel("output/power_consumption_rate.xlsx", header = None, index = False)
os.remove("output/power_consumption_rate.csv")



