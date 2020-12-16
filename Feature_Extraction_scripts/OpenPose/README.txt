OPENPOSE JSON EXTRACTION FROM VIDEO FRAMES

To extract OpenPose json dataset from video frames use extract_openpose_json.py file. The installation of OpenPose is currently made in mil11 server (mil-11l.main.ad.rit.edu) and the path is /shared/kgcoe-research/mil/sign_language_review/installations/openpose/openpose

The json extraction script extract_openpose_json.py need to be placed in the above path (/shared/kgcoe-research/mil/sign_language_review/installations/openpose/openpose) and run.

Need to run "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64" or add in bash profile if OpenPose extraction fails.


OPENPOSE FEATURE EXTRACTION 

To extract features from OpenPose json file use script openPose_feature.py.

The script will generate six folders namely - before_smoothing_no_normalisation, before_smoothing_old_normalisation, before_smoothing_new_normalisation, after_smoothing_no_normalization, after_smoothing_old_normalization, after_smoothing_new_normalization

