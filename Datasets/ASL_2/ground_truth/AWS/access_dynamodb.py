import boto3
import pdb

pdb.set_trace()
dynamodb = boto3.resource("dynamodb", region_name="us-east-2", endpoint_url="https://us-east-2.console.aws.amazon.com/dynamodb/home?region=us-east-2",
                          aws_access_key_id="AKIA37BBQF3Q4RGHO443", aws_secret_access_key="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")

table = dynamodb.Table('ASLingDB_dynamodb')

response = table.scan()