import pandas as pd
import matplotlib.pyplot as plt

files=["grid_0_3x4_i200_multiagent_20240501-1756381714557398.909534-0_emission",
"grid_0_3x4_i200_multiagent_20240501-1756381714557398.9184172-0_emission",
"grid_0_3x4_i200_multiagent_20240501-1756381714557398.9214733-0_emission",
"grid_0_3x4_i200_multiagent_20240501-1756381714557398.9409337-0_emission",
"grid_0_3x4_i200_multiagent_20240501-1756391714557399.015054-0_emission",
"grid_0_3x4_i200_multiagent_20240501-1756391714557399.0168405-0_emission",
"grid_0_3x4_i200_multiagent_20240501-1756391714557399.0347862-0_emission",
"grid_0_3x4_i200_multiagent_20240501-1756391714557399.1261826-0_emission"]
for i in files:
    file=i
    veh="cav_0"
    FILE_PATH = ("/home/g/software/pycharm/projects/flow-SEU/workplace/final/3*4路网/Co_PPO/data/"+
                file+".csv")
    OUTPUT_PATH = '/home/g/software/pycharm/projects/flow-SEU/workplace/final/3*4路网/Co_PPO/data'

    file_path=FILE_PATH
    raw_df = pd.read_csv(file_path)

    df_cav = (raw_df[raw_df["id"].apply(lambda x: x[0:5] == veh)])
    x4 = df_cav['x']
    y4 = df_cav['y']

    fig5 = plt.figure(figsize=(6, 4.8))
    plt.plot(x4, y4 ,c="b")
    xl = 'x'
    yl = "y"
    plt.xlabel(xl)
    plt.ylabel(yl,rotation = 0)
    # plt.show()
    fig5.savefig(OUTPUT_PATH+ '/' +file + '.png')

# df_0 = (raw_df[raw_df["id"].apply(lambda x: x[0:7] == "human_0")])
# x = df_0['time']
# y = df_0['speed']
#
# df_3 = (raw_df[raw_df["id"].apply(lambda x: x[0:7] == "human_3")])
# x1 = df_3['time']
# y1 = df_3['speed']
#
# df_8 = (raw_df[raw_df["id"].apply(lambda x: x[0:7] == "human_8")])
# x2 = df_8['time']
# y2 = df_8['speed']
#
# df_cav = (raw_df[raw_df["id"].apply(lambda x: x[0:5] == "cav_0")])
# x3 = df_cav['time']
# y3 = df_cav['speed']

# fig1 = plt.figure(figsize=(6, 4.8))
# plt.plot(x, y,c="b")
# xl = 'time'
# yl = "velocity of human_0"
# plt.xlabel(xl)
# plt.ylabel(yl)
# plt.axis([0, 100, 0, 30])  # X、Y轴区间
# fig1.savefig(OUTPUT_PATH+ '/' +yl + '.png')
#
# fig2 = plt.figure(figsize=(6, 4.8))
# plt.plot(x1, y1,c="b")
# xl = 'time'
# yl = "velocity of human_1"
# plt.xlabel(xl)
# plt.ylabel(yl)
# plt.axis([0, 200, 0, 30])
# fig2.savefig(OUTPUT_PATH+ '/' +yl + '.png')
#
# fig3 = plt.figure(figsize=(6, 4.8))
# plt.plot(x2, y2 ,c="b")
# xl = 'time'
# yl = "velocity of human_2"
# plt.xlabel(xl)
# plt.ylabel(yl)
# plt.axis([0, 150, 0, 30])
# plt.show()
# fig3.savefig(OUTPUT_PATH+ '/' +yl + '.png')
#
# fig4 = plt.figure(figsize=(6, 4.8))
# plt.plot(x3, y3 ,c="b")
# xl = 'time'
# yl = "velocity of icv"
# plt.xlabel(xl)
# plt.ylabel(yl)
# plt.axis([0, 400, 0, 30])
# plt.show()
# fig4.savefig(OUTPUT_PATH+ '/' +yl + '.png')

