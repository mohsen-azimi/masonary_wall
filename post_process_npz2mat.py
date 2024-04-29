import pickle
import scipy.io as sio
import numpy as np

# Load the .pkl file

file_path = 'input_output/outputs/wood/N882A6_ch2_main_20221012110243_20221012110912.mp4_tracks_and_visibility.npz'

tracking_data = np.load(file_path, allow_pickle=True)




# Save the data to a .mat file
file_path = file_path.replace(".npz",".mat")
sio.savemat(file_path, {'tracking_data': tracking_data})
