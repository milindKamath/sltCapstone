Tejaswini Ananthanarayana

# Configure aws cli on terminal:
	aws configure (Enter)
  Type Access key, secret key, region when prompted

# Print the configuration on terminal:
	aws configure list

# Download s3 data to local computer using AWS cli on terminal
	 aws s3 sync s3://sign-speak-data-lake/ASLing /shared/kgcoe-research/mil/sign_language_review/Datasets/Original_frames/ASl_2

