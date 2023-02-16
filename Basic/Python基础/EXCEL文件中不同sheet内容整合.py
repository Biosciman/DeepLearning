'''
EXCEL文件中不同sheet内容整合
每张sheet中Name有重复
'''

import openpyxl
import xlrd
import numpy as np
import pandas as pd

filename1 = '1号.xlsx'
filename2 = '2号.xlsx'
filename3 = '1号_2号_Merged_file.xlsx'

# 获取每个xlse的title
wb = openpyxl.load_workbook(filename3).worksheets
SheetTitle = []
for i in wb:
    SheetTitle.append(i.title)
print(SheetTitle)

book = xlrd.open_workbook(filename3)

for i in range(len(SheetTitle)):
    sheet = book.sheets()[i]
    name = np.array([x.value for x in sheet.col(1, start_rowx=1)])
    area = np.array([x.value for x in sheet.col(0, start_rowx=1)])
    df1 = pd.DataFrame(name, columns=['Name'])
    df2 = pd.DataFrame(area, columns=[SheetTitle[i]])
    df3 = pd.concat([df1, df2], axis=1).values.tolist()
    df1 = df1.values.tolist()
    exec('{} = df3'.format(SheetTitle[i]))
    exec('{} = df1'.format(SheetTitle[i] + "_name"))

SheetTitle_name = []
for i in SheetTitle:
    SheetTitle_name.append(i+'_name')

# 获取所有Name的list，并去除重复项
TotalName = []
for i in SheetTitle_name:
    print(i)
    for j in eval(i):
        TotalName.append(j[0])

TotalName = list(set(TotalName))
# print(TotalName)
print(len(TotalName))

# 获取每个name在不同sheet中的最大个数
FinalCountList = []
for i in TotalName:
    countList = []
    for j in SheetTitle:
        count = 0
        for k in eval(j):
            if i == k[0]:
                count += 1
        countList.append(count)
    FinalCountList.append([i, max(countList)])
# print(FinalCountList)
print(len(FinalCountList))

# 将SheetTtile安装FinalCountList进行sorting
FinalList = []
for j in SheetTitle:
    sorted = []
    for i in FinalCountList:
        count = 0
        for k in eval(j):
            if i[0] == k[0]:
                count += 1
                sorted.append([i[0], k[1]])
        if count < i[1]:
            for _ in range((i[1]-count)):
                sorted.append([i[0], 'NA'])
    exec('{} = sorted'.format(j + "_sorted"))

ColumnTitle = ['Name'] + SheetTitle
print(ColumnTitle)

SheetTitle_sorted = []
for i in SheetTitle:
    SheetTitle_sorted.append(i+'_sorted')
print(SheetTitle_sorted)

finalDataframe = pd.DataFrame(eval(SheetTitle_sorted[0]))
count = 0
for i in SheetTitle_sorted:
    if count != 0:
        dff = pd.DataFrame(eval(i))
        finalDataframe = pd.concat([finalDataframe, dff[1]], axis=1)
    count += 1
finalDataframe.columns = ColumnTitle
print(finalDataframe)
finalDataframe.to_csv('1号_2号_Merged_file.csv')

