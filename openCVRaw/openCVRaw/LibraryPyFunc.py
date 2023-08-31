import cv2
import numpy as np
import csv
print("Program Start")

###################################################
def writeImageToCSV(filename, image):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in image:
            writer.writerow(row)

def write_to_csv_one_by_one(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)
###################################################
print("Github")
def convertToGray(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray

def ProcessFile(filename):
    img = cv2.imread(filename+'.BMP')

    if img is not None and img.shape[2] == 3:
      if img[0, 0, 0] == img[0, 0, 2]:
            print("The image is in RGB color order.")
      else:
            print("The image is in BGR color order.")
    else:
      print("Failed to load the image.")


    print('Image has acquired WxH: '+str(img.shape[1])+' x '+str(img.shape[0]))
    grayImg = convertToGray(img)
    writeImageToCSV('PyhtonGrayImg.csv',grayImg)
    fn = filename+'ConvertedImg'+'.png';
    cv2.imwrite(fn,grayImg)


### Main Func
sPath = "D:\\openCVRaw\\img\\"  
ProcessFile(sPath+'14264320');
print("Program End")