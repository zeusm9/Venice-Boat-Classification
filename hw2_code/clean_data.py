import string
import os
from pathlib import Path

def clean_truth(filename):
    f = open(filename)
    lines = f.readlines()
    new_ground_truth = Path("/new_ground_truth.txt")
    if (not new_ground_truth.exists()):
        new_file = open("new_ground_truth.txt", "w")

        for line in lines:
            splitted_line = line.split(";")
            label = splitted_line[1].replace(" ", "").translate(str.maketrans('', '', string.punctuation)).replace('\n','')
            my_list = os.listdir('train')
            image_paths = os.listdir("test")
            if label in my_list and splitted_line[0] in image_paths:
                new_line = splitted_line[0] + "\t" + splitted_line[1].replace(" ", "").translate(
                        str.maketrans('', '', string.punctuation))
                new_file.write(new_line)
            elif splitted_line[0] in image_paths and splitted_line[1] == "Snapshot Acqua\n":
                new_line = splitted_line[0] + "\tWater\n"
                new_file.write(new_line)
        new_file.close()

def clean_test():
    new_ground_truth = open("new_ground_truth.txt","r")
    lines = new_ground_truth.readlines()
    labels = []
    for line in lines:
        labels.append(line.split("\t")[1].replace("\n",""))
