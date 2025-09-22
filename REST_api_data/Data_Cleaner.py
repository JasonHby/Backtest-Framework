import os, glob

# remove every .csv in the current working directory
for csv_path in glob.glob("*.csv"):
    os.remove(csv_path)
print("All CSVs deleted.")