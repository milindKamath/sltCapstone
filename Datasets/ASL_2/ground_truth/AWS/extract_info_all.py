import pandas as pd
import pdb
import base64
import ASLing_orchestrator_pb2 as pb
import os
import csv

def extract_gt_info(df, writer):
    for i, row in df.iterrows():
        ling = row["LING_ENCODED (B)"]
        ling = base64.b64decode(ling)
        ling_pb = pb.Linguistics()
        ling_pb.ParseFromString(ling)
        caption = ling_pb.annotation.interpretation
        gloss = " ".join([i.gloss for i in ling_pb.annotation.signs])
        video_path = row["IMAGE_REF (S)"]
        video_name = video_path.split("/")[-1]
        print("Video: ", video_name, " with caption: ", caption, " and gloss: ", gloss)
        # pdb.set_trace()
        writer.writerow({"Video_name": video_name, "Caption": caption, "Gloss": gloss, "Video_path": video_path})
        # pdb.set_trace()



if __name__ == "__main__":
    all_csvs = "/home/ta2184/sign_language_review_paper/Misc/AWS/raw_csv"
    write_csv = "/home/ta2184/sign_language_review_paper/Misc/AWS/ASL_2.csv"
    with open(write_csv,'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Video_name", "Caption", "Gloss", "Video_path"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for file in os.listdir(all_csvs):
            # pdb.set_trace()
            full_file_path = os.path.join(all_csvs, file)
            df = pd.read_csv(full_file_path)
            extract_gt_info(df, writer)





# df = pd.read_csv("/home/ta2184/sign_language_review_paper/Misc/AWS/ASLingDB_dynamodb.csv")
#
# for i, row in df.iterrows():
#     pdb.set_trace()
#     ling = row["LING_ENCODED (B)"]
#     ling = base64.b64decode(ling)
#     ling_pb = pb.Linguistics()
#     ling_pb.ParseFromString(ling)
#     caption = ling_pb.annotation.interpretation
#     gloss = " ".join([i.gloss for i in ling_pb.annotation.signs])

