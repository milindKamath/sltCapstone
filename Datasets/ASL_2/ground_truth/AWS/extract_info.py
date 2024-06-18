import pandas as pd
import pdb
import base64
import ASLing_orchestrator_pb2 as pb

df = pd.read_csv("/home/ta2184/sign_language_review_paper/Misc/AWS/ASLingDB_dynamodb.csv")

for i, row in df.iterrows():
    pdb.set_trace()
    ling = row["LING_ENCODED (B)"]
    ling = base64.b64decode(ling)
    ling_pb = pb.Linguistics()
    ling_pb.ParseFromString(ling)
    caption = ling_pb.annotation.interpretation
    gloss = " ".join([i.gloss for i in ling_pb.annotation.signs])

