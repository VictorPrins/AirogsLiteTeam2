import os
import csv

allids = []

to_add = []

with open('submission_modify.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        allids.append(row['aimi_id'])


for i in range(10000):
    id = 'TEST' + str(i).zfill(5)

    if id not in allids:
        to_add.append(id)

print(to_add)