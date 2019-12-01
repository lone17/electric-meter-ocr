"""
testReader.py

Run: python testReader.py --input_folder input_folder --output_file output.csv --data_folder data

Description: init a reader with data folder then process all images from input_folder,
write the results to output.csv with header (filename,number)
"""

import sys
import os
import time

#version 4.1.1.26
import cv2

# Import reader
from readingMeter import Reader

def process(input_folder, output_file, data_folder):
    start_time = time.time()
    print("Start init reader with data from: ", data_folder)
    reader = Reader(data_folder)
    reader.prepare()
    run_time = time.time() - start_time
    print("Finish init in %.2f second" % run_time)

    # Get all images data
    list_file = os.listdir(input_folder)

    with open(output_file, "w") as f:
        f.write("filename,number\n")

        for fname in list_file:
            img = cv2.imread(os.path.join(input_folder, fname))
            number = reader.process(img)
            f.write('%s,%i\n' % (fname, number))

    print("Finish process %i images" % len(list_file))

def process_crop(input_folder, output_file, data_folder):
    start_time = time.time()
    print("Start init reader with data from: ", data_folder)
    reader = Reader(data_folder)
    reader.prepare_crop()
    run_time = time.time() - start_time
    print("Finish init in %.2f second" % run_time)

    # Get all images data
    list_file = os.listdir(input_folder)

    with open(output_file, "w") as f:
        f.write("filename,number\n")

        for fname in list_file:
            img = cv2.imread(os.path.join(input_folder, fname))
            number = reader.crop_and_process(img)
            f.write('%s,%i\n' % (fname, number))

    print("Finish process %i images" % len(list_file))


if __name__=="__main__":

    # Get input parameters
    input_folder = sys.argv[1]
    output_file = sys.argv[2]
    data_folder = sys.argv[3]

    # Run time
    start_time = time.time()
    #try:
    process(input_folder, output_file, data_folder)
    #except:
    #    pass

    run_time = time.time() - start_time
    print("Total run_time = ", run_time)




