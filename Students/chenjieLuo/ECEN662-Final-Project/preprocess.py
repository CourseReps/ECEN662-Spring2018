from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import os


path = "E:/Texas A&M/Estimate and detection/test_set" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
stations = [4847, 14702, 3888, 14929, 4994, 3032, 14792, 3162, 3870, 24126,
4846, 4826, 3030, 4138, 3960, 24110, 13833, 4110, 14910, 54742,
13838, 3811, 24013, 54771, 54740, 3816, 4112, 3062, 4854, 4130,
4725, 14739, 13764, 3759, 4840, 3160, 3889, 3859, 4781, 3013,
3868, 54728, 4136, 14933, 3031, 21510, 3856, 14752, 4131, 3860,
14794, 21514, 14707, 3074, 3067, 4111, 12884, 4139, 3966, 3893,
12916, 4842, 4137, 24018, 3952, 25309, 3928, 14935, 12812, 4109,
3947, 4836, 14936, 3963, 3728, 428, 3820, 4839, 14745, 3055,
4804, 54743, 13744, 13728, 4223, 3102, 4848, 11640, 4808, 14734,
4803, 21504, 24011, 3878, 13739, 3887, 25323, 3847, 4843, 25308,
13752, 23009, 14914, 14740, 3822, 14606, 14787, 3131, 3919, 3733,
14765, 3932, 3758, 12815, 3849, 14913, 3810, 3024, 3170, 3026,
93721, 14939, 3818, 3949, 14605, 14941, 3145, 14940, 3812, 23176,
4128, 4990, 3894, 3144, 3029, 14763, 3940, 4125, 4726, 13939,
13781, 3953, 4789, 3017, 3739, 4113, 4140, 93706, 14710, 14937]
train_set = pd.DataFrame();
for file in files: #遍历文件夹
     if os.path.isdir(path + '/' + file): #判断是否是文件夹，不是文件夹才打开
          test_set = os.listdir(path + '/' + file)
          for test in test_set:
               data = pd.read_csv(path + '/' + file + '/' + test,usecols=['Wban Number',' Avg Temp',' YearMonthDay'])
               if train_set.empty:
                   train_set = data
               else:
                   train_set = train_set.append(data)
train_set = train_set.loc[train_set['Wban Number'].isin(stations)]
train_set = train_set.sort_values(by=[' YearMonthDay'])
train_set.to_csv('training_set.csv')
print(train_set) #打印结果