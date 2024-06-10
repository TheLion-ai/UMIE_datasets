"""Inspect if the jsson file works fine."""

import json

import jsonlines

# json_path = './data/UMIE_filelist.json'
# with open(json_path, 'r') as json_file:
#     lines = json_file.readlines()
#     json_file.seek(0)
#     for line in lines:
#         line = json.loads(line)

#         if len(line['labels'])>0:
#             print(line)

json_path = "./data/kits23.json"
with open(json_path, "r") as json_file:
    json_file.seek(0)
    for line in json_file:
        line = json.loads(line)
        print(line)
