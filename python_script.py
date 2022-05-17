from Helper import  readImages ,FeaturesParameters ,fitModel , processVideo
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import pickle
import sys

input_video = sys.argv[1]
output_video = sys.argv[2]
debug_flag = sys.argv[3]
#
# input_video = './project_video.mp4'
# output_video = './project_video_output.mp4'
# debug_flag=1
train_flag = 'n'


params = FeaturesParameters()
if train_flag.lower() == 'y':
    vehicles = readImages('./Data/vehicles', '*.png')
    non_vehicles = readImages('./Data/non-vehicles', '*.png')
    svc, scaler, fittingTime, accuracy = fitModel(vehicles, non_vehicles, LinearSVC(), StandardScaler(), params)
    print('Fitting time: {} s, Accuracy: {}'.format(fittingTime, accuracy))

    pickle_data = {}
    pickle_file = open('pickle_file', 'wb')
    pickle_data['scaler'] = scaler
    pickle_data['clsf'] = svc
    pickle.dump(pickle_data, pickle_file)
    pickle_file.close()

pickle_file = open('pickle_file', 'rb')
pickle_data = pickle.load(pickle_file)
scaler = pickle_data['scaler']
svc = pickle_data['clsf']
pickle_file.close()

processVideo(input_video, output_video,svc, scaler, params,debug_flag= debug_flag,
             x_start_stop=[0,-1], threshhold=61)