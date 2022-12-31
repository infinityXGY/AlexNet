import os
import csv
import shutil

with open('train.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        if row[1]=='-1':
            # print(row[0])
            shutil.copyfile('train/trainimages/'+row[0],'data_pre/female/'+row[0])
        else:
            shutil.copyfile('train/trainimages/' + row[0], 'data_pre/male/'+row[0])
