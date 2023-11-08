# script made to correct a generated .csv file with unnecessary commas
if __name__ == "__main__":
    dir_s = ''
    dir_d = ''
    with open(dir_s) as infile, open(dir_d, "w") as outfile:
        for line in infile:
            outfile.write(line.replace(",", ""))

