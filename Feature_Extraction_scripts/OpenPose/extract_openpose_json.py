import os
import cv2
import pandas as pd
import subprocess
import pdb

path = '/shared/kgcoe-research/mil/sign_language_review/Datasets/Original_frames/ASl_2/BcTm9JOPQUSRhAokEoGQRO5ONDY2/0AEJgpx9jGDtPlKDL4wa'

csv_path = '/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Models_scripts/Priyanshu/ASL/asl2_preprocessed_iii.csv'

dataset_path = '/shared/kgcoe-research/mil/sign_language_review/Datasets/Original_frames/ASl_2/'

out = '/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Misc/Priyanshu/OpenPose/Lipisha_100_new/openpose-json/'
out_file = '/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Misc/Priyanshu/ASL_complete_dataset_openpose/out/'
json_file = '/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Misc/Priyanshu/ASL_complete_dataset_openpose/json/'

def createDir(dir):  # create new direcroty
    if not os.path.exists(dir):
        os.makedirs(dir)

def extract():
    db.set_trace()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            # print os.path.join(root, name)
            db.set_trace()
            arg1 = os.path.join(root, name)
            rel_dir = os.path.relpath(root, path)

            arg_3 = os.path.join(out_file, rel_dir)
            arg3 = os.path.join(arg_3, name)
            if not os.path.exists(arg3):
                os.makedirs(arg3)
            arg_2 = os.path.join(out, rel_dir)
            arg2 = os.path.join(arg_2, name)
            # arg2 = os.path.join(out,rel_dir)
            if not os.path.exists(arg2):
                os.makedirs(arg2)
            arg1 = arg1 + '/'
            arg2 = arg2 + '/'
            arg3 = arg3 + '/'
            print(arg1)
            print(arg2)
            print(arg3)
            # print(arg3)
            subprocess.check_call(
                ['./build/examples/openpose/openpose.bin', '--image_dir', arg1, '--write_json', arg3, '--display', '0',
                 '--disable_blending', '--write_images', arg2, '--face', '--face_render', '2', '--hand', '--hand_render',
                 '2', '--keypoint_scale', '3'])  # ,'--keypoint_scale','3'
    #        subprocess.check_call(['./build/examples/openpose/openpose.bin', '--image_dir', arg1,  '--write_json', arg3,'--display', '0', '--disable_blending', '--model_pose', 'COCO','--keypoint_scale','3', '--write_images', arg2])
    # cmd = ['./build/examples/openpose/openpose.bin', '--image_dir', arg1,  '--write_json', arg1,'--display', '0', '--disable_blending',  '--write_images', arg2, '--face', '--face_render', '2', '--hand', '--hand_render', '2']
    # response = subprocess.check_output(cmd,
    #         shell=False,
    #         )

def extract_from_csv2():
    pdb.set_trace()
    base_path = '/shared/kgcoe-research/mil/sign_language_review/Datasets/Original_frames/ASL_2_main_folders/'
    data_frame = pd.read_csv(csv_path)
    name = ''
    for value in data_frame['Video_path']:
        directory_name = value.split('/')
        if directory_name[len(directory_name)-2] == 'YyxKDBikPjUx8sBnBpCF9f2Xy6S2':
            name = 'YyxKDBikPjUx8sBnBpCF9f2Xy6S2_2'
        else:
            name= directory_name[len(directory_name)-2]
        out_path = out_file+name+'/'+directory_name[len(directory_name)-1]
        json_path = json_file + name + '/' + directory_name[len(directory_name) - 1]
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            os.makedirs(json_path)
        value = base_path+name+'/'+directory_name[len(directory_name)-1]

        subprocess.check_call(
            ['./build/examples/openpose/openpose.bin', '--image_dir', value, '--write_json', json_path, '--display', '0',
             '--disable_blending', '--write_images', out_path, '--face', '--face_render', '2', '--hand', '--hand_scale_number',
             '6', '--hand_scale_range', '0.4', '--hand_detector', '3','--keypoint_scale', '3'])


def extract_from_csv():
    createDir("./out")
    createDir("./json")
    data_frame = pd.read_csv(csv_path)
    for value in data_frame['Video_name']:
        directory_name = value
        value = dataset_path + value

        out_path = "./out/" + directory_name
        json_path = "./json/" + directory_name

        subprocess.check_call(
            ['./build/examples/openpose/openpose.bin', '--image_dir', value, '--write_json', json_path, '--display', '0',
             '--disable_blending', '--write_images', out_path, '--face', '--face_render', '2', '--hand', '--hand_scale_number',
             '6', '--hand_scale_range', '0.4', '--hand_detector', '3', '--keypoint_scale', '3'])

def extract_sample():
    value = '/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Models_scripts/Priyanshu/ASL/OpenPose_samples/'
    json_path = '/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Models_scripts/Priyanshu/ASL/OpenPose_samples_json/'
    out_path = '/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Models_scripts/Priyanshu/ASL/OpenPose_samples_json/'

    subprocess.check_call(
        ['./build/examples/openpose/openpose.bin', '--image_dir', value, '--write_json', json_path, '--display', '0',
         '--disable_blending', '--write_images', out_path, '--face', '--face_render', '2', '--hand', '--hand_render',
         '2', '--keypoint_scale', '3'])

if __name__ == "__main__":
    extract_from_csv()