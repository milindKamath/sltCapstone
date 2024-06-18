import os
import json
import pdb
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from scipy.signal import wiener
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.ndimage import median_filter
import numpy as np
import cv2
import scipy
import pandas as pd


def createDir(dir):  # create new direcroty
    if not os.path.exists(dir):
        os.makedirs(dir)


def body_reference(data_body):
    # pdb.set_trace()
    points = [0, 1, 8]

    nz_x = [data_body[x * 2] for x in points if data_body[x * 2] != 0.0]
    nz_y = [data_body[x * 2 + 1] for x in points if data_body[x * 2 + 1] != 0.0]

    if len(nz_x) == 0 or len(nz_y) == 0:  # if all zeros
        nz = (0, 0)
    else:
        nz = (sum(nz_x) / len(nz_x), sum(nz_y) / len(
            nz_y))  # average of all non zero values  ....... averaging just 3 points vs averaging all on zero points?

    return nz


def normalize(data, data_body):
    # pdb.set_trace()
    norm_data = [0] * len(data)
    nz = body_reference(data_body)
    for x in range(0, len(data), 2):
        if data[x] != 0.0:
            norm_data[x] = data[x] - nz[0]
            norm_data[x + 1] = data[x + 1] - nz[1]
        else:
            norm_data[x:x + 2] = data[x:x + 2]

    # data = [(data[x] - nz[0], data[x + 1] - nz[1]) if data[x] != 0.0 else data[x:x + 2] for x in range(0, len(data), 2)]  # apply operation to only non-zero values
    # data = sum(data, [])  # list of list to list
    return norm_data


def normalize_ref(data):
    points = [0, 1, 8]

    nz_x = [data[x * 3] for x in points if data[x * 3] != 0.0]
    nz_y = [data[x * 3 + 1] for x in points if data[x * 3 + 1] != 0.0]

    if len(nz_x) == 0 or len(nz_y) == 0:  # if all zeros
        nz = (0, 0)
    else:
        nz = (sum(nz_x) / len(nz_x), sum(nz_y) / len(
            nz_y))  # average of all non zero values  ....... averaging just 3 points vs averaging all on zero points?

    data = [[data[x] - nz[0], data[x + 1] - nz[1], data[x + 2]] if data[x] != 0.0 else data[x:x + 3] for x in
            range(0, len(data), 3)]  # apply operation to only non-zero values
    data = sum(data, [])  # list of list to list
    return data


def size(data):
    # input is a list of length 54
    # output is a size scalar
    # pdb.set_trace()
    pairs = [[0, 1], [1, 8]]  # all pairs for distance computation

    total_dist = 0.0
    for pair in pairs:
        total_dist += distance.euclidean([data[pair[0] * 3], data[(pair[0] * 3) + 1]],
                                         [data[pair[1] * 3], data[(pair[1] * 3) + 1]])
    return total_dist


def rid_of_zero(data):
    return [-999.0 if x == 0.0 else x for x in data]


# Reference json normalization
ref = \
    json.load(open('/shared/kgcoe-research/mil/phonenix_dataset/Akhil/K2V-color-11508.46973_keypoints_new.json', 'r'))[
    'people'][0]['pose_keypoints_2d']  # reference json, reference should have all body points visible
# pdb.set_trace()
norm_ref = normalize_ref(ref)  # normalize reference datac

counter = 0


def  scale_points(norm_points, scale):
    scaled = [0] * len(norm_points)

    for x in range(0, len(norm_points), 2):
        if norm_points[x] != 0.0:
            scaled[x] = norm_points[x] * scale
            scaled[x + 1] = norm_points[x + 1] * scale  # removed multiplication by -1.0
        else:
            scaled[x:x + 2] = norm_points[x:x + 2]
    return scaled


def connectpoints(body_x, body_y, p1, p2):

    x1, x2 = body_x[p1], body_x[p2]
    y1, y2 = body_y[p1], body_y[p2]

    if (x1 != 0 and y1 != 0) or (x2 != 0 and y2 != 0):
        # plt.plot([x1, x2], [y1, y2], 'k-', linewidth=3, marker=".", markersize=15)
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
        plt.annotate(p1, (x1, y1), rot)
        plt.annotate(p2, (x2, y2))
        plt.gca().set_aspect('equal', adjustable='box')


def zero_to_nan(values):
    return [float('nan') if abs(x) == 0 else x for x in values]


def plot_body(body_x, body_y):
    connectpoints(body_x, body_y, 0, 1)
    connectpoints(body_x, body_y, 0, 15)
    connectpoints(body_x, body_y, 0, 16)
    connectpoints(body_x, body_y, 15, 17)
    connectpoints(body_x, body_y, 16, 18)
    connectpoints(body_x, body_y, 2, 3)
    connectpoints(body_x, body_y, 3, 4)
    connectpoints(body_x, body_y, 1, 2)
    connectpoints(body_x, body_y, 1, 5)
    connectpoints(body_x, body_y, 5, 6)
    connectpoints(body_x, body_y, 6, 7)
    connectpoints(body_x, body_y, 1, 8)
    connectpoints(body_x, body_y, 8, 9)
    connectpoints(body_x, body_y, 9, 10)
    connectpoints(body_x, body_y, 10, 11)
    connectpoints(body_x, body_y, 8, 12)
    connectpoints(body_x, body_y, 12, 13)
    connectpoints(body_x, body_y, 13, 14)
    connectpoints(body_x, body_y, 11, 24)
    connectpoints(body_x, body_y, 11, 22)
    connectpoints(body_x, body_y, 22, 23)
    connectpoints(body_x, body_y, 14, 21)
    connectpoints(body_x, body_y, 14, 19)
    connectpoints(body_x, body_y, 19, 20)
def plot_hand(hand_x, hand_y):

    connectpoints(hand_x, hand_y, 0, 1)  # Avoiding to see if plot looks better
    connectpoints(hand_x, hand_y, 1, 2)
    connectpoints(hand_x, hand_y, 2, 3)
    connectpoints(hand_x, hand_y, 3, 4)
    connectpoints(hand_x, hand_y, 0, 5)  # Avoiding to see if plot looks better
    connectpoints(hand_x, hand_y, 5, 6)
    connectpoints(hand_x, hand_y, 6, 7)
    connectpoints(hand_x, hand_y, 7, 8)
    connectpoints(hand_x, hand_y, 0, 9)  # Avoiding to see if plot looks better
    connectpoints(hand_x, hand_y, 9, 10)
    connectpoints(hand_x, hand_y, 10, 11)
    connectpoints(hand_x, hand_y, 11, 12)
    connectpoints(hand_x, hand_y, 0, 13)  # Avoiding to see if plot looks better
    connectpoints(hand_x, hand_y, 13, 14)
    connectpoints(hand_x, hand_y, 14, 15)
    connectpoints(hand_x, hand_y, 15, 16)
    connectpoints(hand_x, hand_y, 0, 17)  # Avoiding to see if plot looks better
    connectpoints(hand_x, hand_y, 17, 18)
    connectpoints(hand_x, hand_y, 18, 19)
    connectpoints(hand_x, hand_y, 19, 20)


def plot_face(face_x, face_y):
    for i in range(16):
        connectpoints(face_x, face_y, i, i + 1)
    for j in range(17, 21, 1):
        connectpoints(face_x, face_y, j, j + 1)
    for k in range(22, 26, 1):
        connectpoints(face_x, face_y, k, k + 1)
    for p in range(27, 30, 1):
        connectpoints(face_x, face_y, p, p + 1)

    for l in range(36, 41, 1):
        connectpoints(face_x, face_y, l, l + 1)
    connectpoints(face_x, face_y, 41, 36)
    for m in range(42, 47, 1):
        connectpoints(face_x, face_y, m, m + 1)
    connectpoints(face_x, face_y, 47, 42)
    for m in range(42, 47, 1):
        connectpoints(face_x, face_y, m, m + 1)
    for n in range(48, 59, 1):
        connectpoints(face_x, face_y, n, n + 1)
    connectpoints(face_x, face_y, 59, 48)
    for o in range(60, 67, 1):
        connectpoints(face_x, face_y, o, o + 1)
    connectpoints(face_x, face_y, 67, 60)
    connectpoints(face_x, face_y, 1, 1)
    connectpoints(face_x, face_y, 31, 32)
    connectpoints(face_x, face_y, 32, 33)
    connectpoints(face_x, face_y, 33, 34)
    connectpoints(face_x, face_y, 34, 35)

def get_x_y_points(points):
    x_points = []
    y_points = []
    counter = 0
    for i in range(0, len(points), 2):
        if int(points[i] != -999) or int(points[i + 1] != -999):
            x = (points[i])
            y = (points[i + 1] * -1)
            x_d = int(points[i] * -1)
            y_d = int(points[i + 1] * -1)

            plt.scatter(x, y, c='black', marker="o", linewidths=5, s=5)  # marker="o"
            plt.plot(x, y)

            x_points.append(x)
            y_points.append(y)

        counter = counter + 1

    return x_points, y_points


def plot_points(points, s_path, a, b):
    pose_keypoints_2d = 50
    hand_left_keypoints_2d = 42
    hand_right_keypoints_2d = 42
    face_keypoints_2d = 140

    for i in range(points.shape[0]):
        plt.figure(figsize=(15, 15))
        body = points[i][:pose_keypoints_2d]
        lft_hand = points[i][pose_keypoints_2d:(pose_keypoints_2d + hand_left_keypoints_2d)]
        rt_hand = points[i][(pose_keypoints_2d + hand_left_keypoints_2d): (
                    pose_keypoints_2d + hand_left_keypoints_2d + hand_right_keypoints_2d)]
        face = points[i][(pose_keypoints_2d + hand_left_keypoints_2d + hand_right_keypoints_2d):]

        min_val = min(np.min(body), np.min(lft_hand), np.min(rt_hand), np.min(face))
        max_val = max(np.max(body), np.max(lft_hand), np.max(rt_hand), np.max(face))


        body_x, body_y = get_x_y_points(body)

        body_x = zero_to_nan(body_x)


        plot_body(body_x, body_y)

        lt_x, lt_y = get_x_y_points(lft_hand)
        plot_hand(lt_x, lt_y)

        rt_x, rt_y = get_x_y_points(rt_hand)
        plot_hand(rt_x, rt_y)

        face_x, face_y = get_x_y_points(face)
        plot_face(face_x, face_y)

        createDir(s_path)

        plt.xlim(a, b)
        plt.ylim(a, b)

    plt.savefig(s_path + str(i) + ".png")

def canonical(median_points):
    pose_keypoints_2d = 50
    hand_left_keypoints_2d = 42
    hand_right_keypoints_2d = 42
    face_keypoints_2d = 140

    after_canonical = [0] * median_points.shape[0]

    for i in range(median_points.shape[0]):
        body = median_points[i][:pose_keypoints_2d]
        lft_hand = median_points[i][pose_keypoints_2d:(pose_keypoints_2d + hand_left_keypoints_2d)]
        rt_hand = median_points[i][(pose_keypoints_2d + hand_left_keypoints_2d): (
                    pose_keypoints_2d + hand_left_keypoints_2d + hand_right_keypoints_2d)]
        face = median_points[i][(pose_keypoints_2d + hand_left_keypoints_2d + hand_right_keypoints_2d):]

        body_norm = normalize(body, body)
        lft_hand_norm = normalize(lft_hand, body)
        rt_hand_norm = normalize(rt_hand, body)
        face_norm = normalize(face, body)

        ref_zero = [norm_ref[x] if x not in [i for i, e in enumerate(body) if e == 0] else 0 for x in range(len(body))]
        scale = size(ref_zero) / size(body)

        body_scaled = scale_points(body_norm, scale)
        lt_hand_scaled = scale_points(lft_hand_norm, scale)
        rt_hand_scaled = scale_points(rt_hand_norm, scale)
        face_scaled = scale_points(face_norm, scale)

        after_canonical[i] = body_scaled + lt_hand_scaled + rt_hand_scaled + face_scaled

    return after_canonical

def normalize_canonical(after_canonical):

    after_canonical_norm = [0] * after_canonical.shape[1]

    after_canonical = np.transpose(after_canonical) #(274, 114)
    for x in range(0, (after_canonical.shape[0])):
        min_can = np.min(after_canonical[x])

        after_canonical_norm[x] = after_canonical[x] + abs(min_can)
        max_can = np.max(after_canonical_norm[x])
        if max_can != 0.0:
            after_canonical_norm[x] = [after_canonical_norm[x] / (max_can - np.min(after_canonical_norm[x]))]
        else:
            after_canonical_norm[x] = [after_canonical_norm[x]]
        if np.isnan(after_canonical_norm[x]).all():
            print("data has nan's")

    after_canonical_norm = np.asarray(after_canonical_norm)
    after_canonical_norm = after_canonical_norm.reshape((after_canonical_norm.shape[0], after_canonical_norm.shape[-1]))
    after_canonical_norm = np.transpose(after_canonical_norm)
    return after_canonical_norm

def get_numpy(points, s1_path):
    np.save(s1_path, points)

def new_normalization(points):
    X=[]
    Y=[]
    for x in range(0, (points.shape[0])):
        x_frame = []
        y_frame = []
        for i in range(0, len(points[x]), 2):
            Vx = points[x][i]
            Vy = points[x][i+1]
            x_frame.append(Vx)
            y_frame.append(Vy)
        x_frame = np.asarray(x_frame)
        y_frame = np.asarray(y_frame)
        x_mean = np.mean(x_frame)
        x_deviation = np.std(x_frame)
        x_frame = x_frame-x_mean
        x_frame = x_frame/x_deviation
        X.append(x_frame)

        y_mean = np.mean(y_frame)
        y_deviation = np.std(y_frame)
        y_frame = y_frame - y_mean
        y_frame = y_frame / y_deviation
        Y.append(y_frame)

    after_normalised=[]
    for i in range(len(X)):
        frame=[]
        for j in range(len(X[i])):
            frame.append(X[i][j])
            frame.append(Y[i][j])
        after_normalised.append(frame)
    return after_normalised

def main():
    i = 0
    createDir("./before_smoothing_no_normalisation")
    createDir("./after_smoothing_no_normalisation")
    createDir("./after_smoothing_old_normalization")
    createDir("./after_smoothing_new_normalization")
    createDir("./before_smoothing_old_normalisation")
    createDir("before_smoothing_new_normalisation")
    csv_path = '/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Models_scripts/Priyanshu/ASL/new_final_filtered_csv/asl2_preprocessed_iii_filtered.csv'
    for name in os.listdir('/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Misc/Priyanshu/ASL_complete_dataset_openpose/json/'):
            locs = ['pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d', 'face_keypoints_2d']
            json_path = (os.path.join('/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Misc/Priyanshu/ASL_complete_dataset_openpose/json/', name))

            json_files = sorted(os.listdir(json_path))

            for data_file in json_files:

                data_frame = pd.read_csv(csv_path)
                video_obj = data_frame[data_frame['Video_name'] == data_file]
                if video_obj.empty:
                    continue
                gloss_frames = "".join([i for i in video_obj['gloss_frames']]).split(',')
                i=i+1
                one_file = []
                seq = data_file + '.npy'

                for fr_num in gloss_frames:
                    json_str = fr_num+"_keypoints.json"
                    full_file = os.path.join(json_path + '/' + data_file, json_str)
                    cur_data = json.load(open(full_file, 'r'))
                    cur_data_new = {}
                    all_files = []

                    for loc_no, loc in enumerate(locs):
                        data = []
                        if not len(cur_data['people']) == 0:
                            if len(cur_data['people']) > 1:
                                check_persons = []
                                for x in range(len(cur_data['people'])):
                                    check_persons.append(cur_data['people'][x]['pose_keypoints_2d'][3])
                                correct_person_index = min(range(len(check_persons)),
                                                           key=lambda i: abs(check_persons[i] - 960.0))
                                data = cur_data['people'][correct_person_index][loc]
                            elif len(cur_data['people']) == 1:
                                data = cur_data['people'][0][loc]

                        cur_data_new[loc] = data
                        all_files.append(cur_data_new[loc])

                    # removing body keypoints
                    if len(all_files[0]) > 0:
                        for i in range(24, 45):
                            all_files[0][i] = 0
                        for i in range(57, 75):
                            all_files[0][i] = 0

                    one_file.append(all_files[0] + all_files[1] + all_files[2] + all_files[3])

                one_file = np.asarray(one_file)
                remove_conf_full_raw = []

                for r in range(one_file.shape[0]):
                    remove_conf = []
                    if len(one_file[r]) == 0:
                        # pdb.set_trace()
                        one_file[r] = one_file[r-1]
                    for s in range(0, len(one_file[r]), 3):
                        remove_conf.append(one_file[r][s])
                        remove_conf.append(one_file[r][s + 1])
                    remove_conf_full_raw.append(remove_conf)
                remove_conf_full_raw = np.asarray(remove_conf_full_raw)

                #Smoothing on raw openPose points

                remove_conf_full = np.transpose(remove_conf_full_raw)
                savgol_filter_op = [0] * remove_conf_full.shape[0]
                median_filt_out = [0] * remove_conf_full.shape[0]
                savgol_filter_op_after_median = [0] * remove_conf_full.shape[0]
                for p in range(remove_conf_full.shape[0]):
                     savgol_filter_op[p] = savgol_filter(remove_conf_full[p], 37, 4, mode='nearest')  # 37, 11
                     median_filt_out[p] = medfilt(remove_conf_full[p], 3)
                     savgol_filter_op_after_median[p] = savgol_filter(median_filt_out[p], 37, 4, mode='nearest')

                savgol_filter_op = np.asarray(savgol_filter_op)
                savgol_filter_op = np.transpose(savgol_filter_op)
                #plot_points(savgol_filter_op,s_path+"raw_savgol/",-1,0)

                median_filt_out = np.asarray(median_filt_out)
                median_filt_out = np.transpose(median_filt_out)
                #plot_points(median_filt_out, s_path+"median/",-1,0) #commenting to speed up canonical

                savgol_filter_op_after_median = np.asarray(savgol_filter_op_after_median)
                savgol_filter_op_after_median = np.transpose(savgol_filter_op_after_median)
                #plot_points(savgol_filter_op_after_median, s_path+'/',-1,0) #commenting to speed up canonical

                print('Completed smoothing of raw OpenPose points')

                #Canonical conversion after smoothing of raw openpose points
                after_canonical_after_smoothing = canonical(savgol_filter_op_after_median)
                after_canonical_after_smoothing = np.asarray(after_canonical_after_smoothing)
                get_numpy(after_canonical_after_smoothing, "./after_smoothing_no_normalisation/"+seq)

                # Canonical conversion before smoothing of raw openpose points
                after_canonical_before_smoothing = canonical(remove_conf_full_raw)
                after_canonical_before_smoothing = np.asarray(after_canonical_before_smoothing)
                get_numpy(after_canonical_before_smoothing, "./before_smoothing_no_normalisation/" + seq)
                #plot_points(after_canonical, s_path+"canonical_savgol_median/",-170,170)  # only this

                print('Completed canonical conversion')

                #old Normalization (min-max) on after canonical, after smoothing points
                after_canonical_old_nrm = normalize_canonical(after_canonical_after_smoothing)  # save this
                get_numpy(after_canonical_old_nrm, "./after_smoothing_old_normalization/"+seq)

                #old Normalization (min-max) on after canonical, before smoothing points
                after_canonical_before_smoothing_old_nrm = normalize_canonical(after_canonical_before_smoothing)
                get_numpy(after_canonical_before_smoothing_old_nrm, "./before_smoothing_old_normalisation/" + seq)

                #new Normalization (standardization) on after canonical, after smoothing points
                after_canonical_new_norm = new_normalization(after_canonical_after_smoothing)  # save this
                after_canonical_new_norm = np.asarray(after_canonical_new_norm)
                get_numpy(after_canonical_new_norm, "./after_smoothing_new_normalization/"+seq)

                #new Normalization (standardization) on after canonical, before smoothing points
                after_canonical_before_smoothing_new_norm = new_normalization(after_canonical_before_smoothing)  # save this
                after_canonical_before_smoothing_new_norm = np.asarray(after_canonical_before_smoothing_new_norm)
                get_numpy(after_canonical_before_smoothing_new_norm, "./before_smoothing_new_normalisation/" + seq)

                print('Completed normalization')

                print("Completed feature set generation for video id "+ seq[:-4])

if __name__ == '__main__':
    main()
