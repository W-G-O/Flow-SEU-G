import sys


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def getResults(PATH):
    import os
    import numpy as np
    import pandas as pd
    # 当前目录
    dataPath = "/home/g/software/pycharm/projects/flow-SEU/workplace/final/"+PATH+"/data"
    TT = []
    DELAY = []
    FUEL = []
    CO2CO2 = []
    COLLISION = []
    VV = []

    # 获取当前目录下的所有文件
    files = [os.path.join(dataPath, file) for file in os.listdir(dataPath)]

    # 遍历文件列表，输出文件名
    for file in files:
        df = pd.read_csv(file,low_memory=False)

        veh_ids = []

        for id in df["id"]:
            if id =="cav_0":
                veh_ids.append(id)
        T = []  # 所有车辆的行驶时间
        Delay = []  # 所有车辆的延误
        Fuel = []  # 所有车辆的燃油消耗
        CO2 = []  # 所有车辆的CO2排放
        Collision = []  # 所有车辆的可能冲突的时间
        S = []  # 所有车辆的行驶距离

        for veh_id in veh_ids:
            veh_attribute = df.loc[df["id"] == veh_id]
            time = list(veh_attribute["time"])

            ############################## 行驶时间（s）###########################
            time0 = time[-1] - time[0]  # 该车辆总行驶时间
            T.append(time0)

            v = list(veh_attribute["speed"])
            s = 0
            fuel = 0
            co2 = 0
            collision = 0
            for i in range(len(time) - 1):
                v0 = (v[i] + v[i + 1]) / 2  # 平均速度（m/s）
                t0 = (time[i + 1] - time[i])  # 行驶时间（s）
                s0 = v0 * t0  # 行驶距离（m）
                s += s0  # 该车辆总行驶距离
                alpha = 0
                if 0 <= v0 * 3.6 <= 5:
                    alpha = 17  # L/100km
                elif 5 < v0 * 3.6 <= 10:
                    alpha = 14  # L/100km
                elif 10 < v0 * 3.6 <= 15:
                    alpha = 12  # L/100km
                elif 15 < v0 * 3.6 <= 20:
                    alpha = 10.5  # L/100km
                elif 20 < v0 * 3.6 <= 25:
                    alpha = 9.3  # L/100km
                elif 25 < v0 * 3.6 <= 30:
                    alpha = 8.25  # L/100km
                elif 30 < v0 * 3.6 <= 35:
                    alpha = 7.4  # L/100km
                elif 35 < v0 * 3.6 <= 40:
                    alpha = 6.65  # L/100km
                elif 40 < v0 * 3.6 <= 45:
                    alpha = 6  # L/100km
                elif 45 < v0 * 3.6 <= 50:
                    alpha = 5.5  # L/100km
                elif 50 < v0 * 3.6 <= 55:
                    alpha = 5.1  # L/100km
                elif 55 < v0 * 3.6 <= 60:
                    alpha = 4.8  # L/100km
                elif 60 < v0 * 3.6 <= 65:
                    alpha = 4.6  # L/100km
                elif 65 < v0 * 3.6 <= 70:
                    alpha = 4.3  # L/100km
                elif 70 < v0 * 3.6 <= 75:
                    alpha = 4.3  # L/100km
                elif 75 < v0 * 3.6 <= 80:
                    alpha = 4.5  # L/100km
                elif 80 < v0 * 3.6 <= 85:
                    alpha = 4.8  # L/100km
                elif 85 < v0 * 3.6 <= 90:
                    alpha = 5.2  # L/100km
                elif 90 < v0 * 3.6 <= 95:
                    alpha = 5.7  # L/100km
                elif 95 < v0 * 3.6 <= 100:
                    alpha = 6.3  # L/100km
                elif 100 < v0 * 3.6 <= 105:
                    alpha = 7.0  # L/100km
                elif 105 < v0 * 3.6 <= 110:
                    alpha = 7.9  # L/100km
                elif 110 < v0 * 3.6 <= 115:
                    alpha = 8.9  # L/100km
                elif 120 < v0 * 3.6 <= 125:
                    alpha = 10.2  # L/100km

                fuel0 = s0 / 1000 / 100 * alpha
                fuel += fuel0  # 该车辆总油耗
                co20 = fuel0 * 2.254 * 1000
                co2 += co20  # 该车辆总CO2排放

            leader_id = list(veh_attribute["leader_id"])
            for i in range(len(leader_id) - 1):
                if type(leader_id[i]) is str:
                    Owner_v = list(veh_attribute["speed"])
                    Leader_v = list(veh_attribute["leader_rel_speed"])
                    Leader_dis = list(veh_attribute["headway"])
                    Time = list(veh_attribute["time"])
                    owner_v = Owner_v[i]
                    leader_v = Leader_v[i]
                    leader_dis = Leader_dis[i]
                    if owner_v > leader_v and leader_dis / (owner_v - leader_v) < 3.00:
                        collision += (Time[i + 1] - Time[i])

            S.append(s)  # 该车辆总行驶距离（m）

            ############################## 延误（s）###########################
            delay = time0 - s / 30
            Delay.append(delay)

            ############################## 油耗（L）###########################
            Fuel.append(fuel)  # 该车辆总油耗（L）

            ############################## CO2排放（g）###########################
            CO2.append(co2)  # 该车辆总CO2排放（g）

            ############################## 冲突时间（s）###########################
            Collision.append(collision)


        #先计算每辆车的，再计算平均值
        # mean_t=[]
        # mean_delay = []
        # mean_collision=[]
        # mean_fuel=[]
        # mean_co2=[]
        # for i in range(len(S)):
        #     mean_t.append(T[i]/(S[i]/1000))
        #     mean_delay.append(Delay[i]/(S[i]/1000))
        #     mean_collision.append(Collision[i]/(S[i]/1000))
        #     mean_fuel.append(Fuel[i]/(S[i]/1000/100))
        #     mean_co2.append(CO2[i]/(S[i]/1000))
        #
        # mean_V = np.sum(S) / np.sum(T)
        # mean_T = np.mean(mean_t)
        # mean_Delay = np.mean(mean_delay)
        # mean_Collision = np.mean(mean_collision)
        # mean_Fuel = np.mean(mean_fuel)
        # mean_CO2 = np.mean(mean_co2)

        #用总的除以总的
        mean_V = np.sum(S) / np.sum(T)
        mean_T = np.sum(T) / (np.sum(S)/1000)
        mean_Delay = np.sum(Delay) / (np.sum(S)/1000)
        mean_Collision = np.sum(Collision) / (np.sum(S)/1000)
        mean_Fuel = np.sum(Fuel) / (np.sum(S)/1000/100)
        mean_CO2 = np.sum(CO2) / (np.sum(S)/1000)

        VV.append(mean_V)
        TT.append(mean_T)
        DELAY.append(mean_Delay)
        COLLISION.append(mean_Collision)
        FUEL.append(mean_Fuel)
        CO2CO2.append(mean_CO2)



    print("*******************************",PATH,"的结果为：*****************************")
    print("*******************************",PATH,"的结果为：*****************************")
    print("*******************************",PATH,"的结果为：*****************************")
    print("智能网联车辆平均速度：", VV)
    print("智能网联车辆平均行驶时间：",TT)
    print("智能网联车辆平均延误：",DELAY)
    print("智能网联车辆可能碰撞平均时间：", COLLISION)
    print("智能网联车辆平均燃油消耗：",FUEL)
    print("智能网联车辆CO2平均排放量：", CO2CO2)


    mean_VV = np.mean(VV)
    mean_TT=np.mean(TT)
    mean_DELAY = np.mean(DELAY)
    mean_COLLISION = np.mean(COLLISION)
    mean_FUEL = np.mean(FUEL)
    mean_CO2CO2 = np.mean(CO2CO2)


    print("**********************最后结果********************************")
    print("智能网联车辆车辆平均速度：", mean_VV, "m/s")
    print("智能网联车辆车辆平均行驶时间：", mean_TT, "s/km")
    print("智能网联车辆车辆平均延误：", mean_DELAY, "s/km")
    print("智能网联车辆车辆可能碰撞平均时间：", mean_COLLISION, "s/km")
    print("智能网联车辆车辆平均燃油消耗：", mean_FUEL, "L/100km")
    print("智能网联车辆车辆CO2平均排放量：", mean_CO2CO2, "g/km")


    print("*************************************************************************")
    print("*************************************************************************")
    print("*************************************************************************")
    print()
    print()
    print()
    print()
    print()
    print()

PATHS=["1*1路网/Co_PPO","1*1路网/fixedtime_cav","1*1路网/In_PPO","1*1路网/RL_HV",
       "1*6路网/Co_PPO","1*6路网/fixedtime_cav","1*6路网/In_PPO","1*6路网/RL_HV",
       "3*4路网/Co_PPO","3*4路网/fixedtime_cav","3*4路网/In_PPO","3*4路网/RL_HV",]
sys.stdout = Logger(filename="/home/g/software/pycharm/projects/flow-SEU/workplace/final/" + 'icv_result.txt',
                    stream=sys.stdout)
for PATH in PATHS:
    getResults(PATH)



