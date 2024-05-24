import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = ("/home/g/software/pycharm/projects/flow-SEU/workplace/final/3*4路网/fixedtime_cav/data（复件）/"+
             "1" + ".csv")
OUTPUT_PATH = '/home/g/software/pycharm/projects/flow-SEU/workplace/final/3*4路网/fixedtime_cav/data（复件）'
VEH_LIST = [str]


def compute(file_path, type_veh: str, method: str):
    """
    type_veh legal: human or cav ids
    method legal: velocity, headway, 
    """

    raw_df = pd.read_csv(file_path)
    df = raw_df[raw_df["id"] == type_veh]
    res = None
    # draw average velocity
    if method == 'velocity':
        df_velocity = df['speed']
        res = df_velocity.describe()

    if method == 'headway':
        df_headway = df['headway']
        res = df_headway.describe()
    print(res)

    return 0


def plot_v_t(file_path, type_veh: str, method:str):

    raw_df = pd.read_csv(file_path)
    df = raw_df[raw_df["id"].apply(lambda x: x[0:7] == type_veh)]
    grouped_by_ids_df = df.groupby(by=['id'])
    fig = plt.figure(figsize=(6, 4.8))

    for name, sub_df in grouped_by_ids_df:
        x = None
        y = None
        if method == 'v-t':
            x = sub_df['time']
            y = sub_df['speed']
            plt.axis([0, 500, 0, 30])

        if method == 'x-y':
            x = sub_df['x']
            y = sub_df['y']
        plt.plot(x, y)

    if method == 'v-t':
        xl = 'time of simulation'
        yl = 'velocity of ' + "icv"
        plt.xlabel(xl)
        plt.ylabel(yl)

    if method == 'x-y':
        xl = 'x'
        yl = "y"
        plt.xlabel(xl)
        plt.ylabel(yl)

    plt.show()
    fig.savefig(OUTPUT_PATH+ '/'  + type_veh+'_'+method + '.png')
    return 0


'''compute(FILE_PATH,
        type_veh='cav_0',
        method='velocity')

compute(FILE_PATH,
        type_veh='cav_0',
        method='headway')'''


# plot_v_t(FILE_PATH,
#          type_veh='cav',method='x-y')
#
# plot_v_t(FILE_PATH,
#          type_veh='hum',method='x-y')
#
# plot_v_t(FILE_PATH,
#          type_veh='cav',method='v-t')

# plot_v_t(FILE_PATH,
#          type_veh='human_0',method='v-t')

file_path=FILE_PATH
raw_df = pd.read_csv(file_path)
df_0 = (raw_df[raw_df["id"].apply(lambda x: x[0:7] == "human_0")])
x = df_0['time']
y = df_0['speed']

df_3 = (raw_df[raw_df["id"].apply(lambda x: x[0:7] == "human_3")])
x1 = df_3['time']
y1 = df_3['speed']

df_8 = (raw_df[raw_df["id"].apply(lambda x: x[0:7] == "human_8")])
x2 = df_8['time']
y2 = df_8['speed']

df_cav = (raw_df[raw_df["id"].apply(lambda x: x[0:5] == "cav_0")])
x3 = df_cav['time']
y3 = df_cav['speed']

df_cav = (raw_df[raw_df["id"].apply(lambda x: x[0:5] == "cav_0")])
x4 = df_cav['x']
y4 = df_cav['y']

fig1 = plt.figure(figsize=(6, 4.8))
plt.plot(x, y,c="b")
xl = 'time(s)'
yl = "velocity_human_0(km/h)"
plt.xlabel(xl)
plt.ylabel(yl)
plt.axis([0, 100, 0, 60])  # X、Y轴区间
fig1.savefig(OUTPUT_PATH+ '/' +"velocity_human_0" + '.png')

fig2 = plt.figure(figsize=(6, 4.8))
plt.plot(x1, y1,c="b")
xl = 'time(s)'
yl = "velocity_human_1(km/h)"
plt.xlabel(xl)
plt.ylabel(yl)
plt.axis([0, 200, 0, 60])
fig2.savefig(OUTPUT_PATH+ '/' +"velocity_human_1" + '.png')

fig3 = plt.figure(figsize=(6, 4.8))
plt.plot(x2, y2 ,c="b")
xl = 'time(s)'
yl = "velocity_human_2(km/h)"
plt.xlabel(xl)
plt.ylabel(yl)
plt.axis([0, 150, 0, 60])
plt.show()
fig3.savefig(OUTPUT_PATH+ '/' +"velocity_human_2" + '.png')

fig4 = plt.figure(figsize=(6, 4.8))
plt.plot(x3, y3 ,c="b")
xl = 'time(s)'
yl = "velocity_icv_0(km/h)"
plt.xlabel(xl)
plt.ylabel(yl)
plt.axis([0, 400, 0, 60])
plt.show()
fig4.savefig(OUTPUT_PATH+ '/' +"velocity_icv_0"+ '.png')

# fig5 = plt.figure(figsize=(6, 4.8))
# plt.plot(x4, y4 ,c="b")
# xl = 'x'
# yl = "y"
# plt.xlabel(xl)
# plt.ylabel(yl,rotation = 0)
# plt.show()
# fig5.savefig(OUTPUT_PATH+ '/' +yl + '.png')