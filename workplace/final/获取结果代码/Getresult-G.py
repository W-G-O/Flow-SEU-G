import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PATH_1 = "/home/g/software/pycharm/projects/flow-SEU/workplace/final/1*1路网/baseline/data/"
PATH_2="fixedtime_cav_20240504-1611411714810301.952846-0_emission.csv"
PATH=PATH_1+PATH_2

df = pd.read_csv(PATH)

veh_ids=[]

for id in df["id"]:
    if id not in veh_ids:
        veh_ids.append(id)
T=[]#所有车辆的行驶时间
Delay=[]#所有车辆的延误
Fuel=[]#所有车辆的燃油消耗
CO2=[]#所有车辆的CO2排放
Collision=[]#所有车辆的可能冲突的时间
S=[]#所有车辆的行驶距离


for veh_id in veh_ids:
    veh_attribute=df.loc[df["id"] == veh_id]
    time=list(veh_attribute["time"])

    ############################## 行驶时间（s）###########################
    time0=time[-1]-time[0] #该车辆总行驶时间
    T.append(time0)

    v=list(veh_attribute["speed"])
    s=0
    fuel=0
    co2=0
    collision = 0
    for i in range(len(time)-1):
        v0=(v[i]+v[i+1])/2#平均速度（m/s）
        t0=(time[i+1]-time[i]) #行驶时间（s）
        s0=v0*t0 #行驶距离（m）
        s += s0 #该车辆总行驶距离
        alpha=0
        if 0 <= v0*3.6 <= 5:
            alpha=17 #L/100km
        elif 5 < v0*3.6 <= 10:
            alpha=14 #L/100km
        elif 10 < v0*3.6 <= 15:
            alpha=12 #L/100km
        elif 15 < v0*3.6 <= 20:
            alpha=10.5 #L/100km
        elif 20 < v0*3.6 <= 25:
            alpha=9.3 #L/100km
        elif 25 < v0*3.6 <= 30:
            alpha=8.25 #L/100km
        elif 30 < v0*3.6 <= 35:
            alpha=7.4 #L/100km
        elif 35 < v0*3.6 <= 40:
            alpha=6.65 #L/100km
        elif 40 < v0*3.6 <= 45:
            alpha=6 #L/100km
        elif 45 < v0*3.6 <= 50:
            alpha=5.5 #L/100km
        elif 50 < v0*3.6 <= 55:
            alpha=5.1 #L/100km
        elif 55 < v0*3.6 <= 60:
            alpha=4.8 #L/100km
        elif 60 < v0*3.6 <= 65:
            alpha=4.6 #L/100km
        elif 65 < v0*3.6 <= 70:
            alpha=4.3 #L/100km
        elif 70 < v0*3.6 <= 75:
            alpha=4.3 #L/100km
        elif 75 < v0*3.6 <= 80:
            alpha=4.5 #L/100km
        elif 80 < v0*3.6 <= 85:
            alpha=4.8 #L/100km
        elif 85 < v0*3.6 <= 90:
            alpha=5.2 #L/100km
        elif 90 < v0*3.6 <= 95:
            alpha=5.7 #L/100km
        elif 95 < v0*3.6 <= 100:
            alpha=6.3 #L/100km
        elif 100 < v0*3.6 <= 105:
            alpha=7.0 #L/100km
        elif 105 < v0*3.6 <= 110:
            alpha=7.9 #L/100km
        elif 110 < v0*3.6 <= 115:
            alpha=8.9 #L/100km
        elif 120 < v0*3.6 <= 125:
            alpha=10.2 #L/100km

        fuel0=s0/1000/100*alpha
        fuel+=fuel0 #该车辆总油耗
        co20=fuel0*2.254*1000
        co2+=co20 #该车辆总CO2排放

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

    S.append(s) #该车辆总行驶距离（m）

    ############################## 延误（s）###########################
    delay=time0-s/30
    Delay.append(delay)

    ############################## 油耗（L）###########################
    Fuel.append(fuel)  # 该车辆总油耗（L）

    ############################## CO2排放（g）###########################
    CO2.append(co2)  # 该车辆总CO2排放（g）

    ############################## 冲突时间（s）###########################
    Collision.append(collision)


mean_T=np.mean(T)
mean_Delay=np.mean(Delay)
mean_Fuel=np.mean(Fuel)
mean_CO2=np.mean(CO2)
All_Collision=np.sum(Collision)
V=np.sum(S)/np.sum(T)
print("车辆平均行驶时间：",mean_T,"s")
print("车辆平均延误：",mean_Delay,"s")
print("车辆平均燃油消耗：",mean_Fuel,"L")
print("车辆平均CO2排放：",mean_CO2,"g")
print("可能冲突总时间：",All_Collision,"s")
print("车辆平均速度：",V,"m/s")


