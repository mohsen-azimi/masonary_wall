import json
import datetime
import numpy as np
from matplotlib import pyplot as plt





# load json file from ./results/marker_detection.json
with open('./results/marker_detection.json', 'r') as f:
    json_data = json.load(f)

my_id = 744
temp_history = []
temp_timeline = []
# print(temp_history.shape)
time_format = '%H:%M:%S'  # Set the time format

# loop over the json data and extract the
for i in range(len(json_data)):
    file_name = json_data[i]['file_name']
    date = json_data[i]['date']
    time = json_data[i]['time']
    corners = json_data[i]['corners']
    ids = json_data[i]['ids']
    dict = json_data[i]['dict']


    if type(ids) is not int:
        if my_id in ids:
            print("Found Marker")
            id_index = ids.index(my_id)
            corner = corners[id_index]
            # print(corner)

            coords = np.array(corner)
            x = coords[:, 0]
            y = coords[:, 1]

            print(time, type(time))

            temp_history.append(y[0])
            temp_timeline.append(datetime.datetime.strptime(time, time_format))

# plot the history of the marker
plt.plot(temp_timeline, temp_history)
plt.show()
