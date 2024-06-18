The joeyNMT code expects input file in a specific format. The generate_joeynmt_dataset.py can be used to generate joeynmt transformer consumable input files (train, test and dev).

The py file has four different functions-

gen_asl() -> This reads the preprocessed csv file for new ASL dataset creates a train and test file.

gen_csl() -> This function reads train_1792_split_sentences.csv and test_448_split_sentences.csv and creates test and train file

gen_gsl() -> reads the csv and generates test, train and dev file

gen_gsl_kmeans() -> This functions reads the GSL test, train and dev csvs and also reads k-means csv file to create means data file in joeynmt format.

The code is little redundant. But due to multiple reading of CSV and different directory structure for each dataset (GSL, CSL and ASL) having different functions for each type helped a lot.

The path variable in each function points to dataset path containing the npy files.

The dev_corpus, test_corpus and train_corpus refer to dev, test and train CSVs.

USAGE Example:

python3 generate_joeyNMT_dataset.py CSL [path to CSL test, train and devfile]   

For usage help run --> python3 generate_joeyNMT_dataset.py -h

 