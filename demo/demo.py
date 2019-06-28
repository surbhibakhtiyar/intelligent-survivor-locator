# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import os
import numpy as np
def ConvertBlue(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	#for (lower, upper) in boundaries:
		#lower = np.array(lower, dtype = "uint8")
		#upper = np.array(upper, dtype = "uint8")
 
		# find the colors within the specified boundaries and apply
		# the mask
	#lower = np.array([100,50,50])
	lower = np.array([100,150,0])
	upper = np.array([140,255,255])
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
	#cv2.imshow("images",output)
	#cv2.waitKey(0)
	return(output)
	
# construct the argument parse and parse the arguments

def imageCompare(first, second):
        # load the two input images
        imageA = cv2.imread(first)
        for i in range(255):
                imageA[np.where((imageA == [i,i,i]).all(axis = 2))] = [255, 0, 0]
        #cv2.imshow("imageA Modified",imageA)
        imageB = cv2.imread(second)
        outputA = ConvertBlue(imageA)
        outputB = ConvertBlue(imageB)


        # convert the images to grayscale
        grayA = cv2.cvtColor(outputA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(outputB, cv2.COLOR_BGR2GRAY)

        #cv2.imshow("intermediate2", grayB- grayA)
        #cv2.waitKey(0)



        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM: {}".format(score))

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU)[1]
        # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # loop over the contours
        # for c in cnts:
        #                 # compute the bounding box of the contour and then draw the
        #                 # bounding box on both input images to represent where the two
        #                 # images differ
        #                 (x, y, w, h) = cv2.boundingRect(c)
        #                 cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #                 cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
         
        # show the output images
        #cv2.imshow("Original", imageA)
        #cv2.imshow("Modified", imageB)
        #cv2.imshow("Diff", diff)
        #cv2.imshow("Thresh", thresh)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite('output2.png',diff)
        from PIL import Image
        import sys

        im = Image.open("output2.png")
        im = im.convert('RGB')
        pixdata = im.load()
        print(im.size)
        #print(pixdata[x,y])
        #for pixel in im.getdata():

        for y in range(im.size[1]):
            for x in range(im.size[0]):
                '''RGB= im.getpixel((x,y))
                R,G,B = RGB
                if G!= 255:
                    R = 255
                im.setpixel((x,y))=(R,G,B)'''
                #print(RGB)
                if pixdata[x, y] == (0, 0, 0):
                    if x+10<im.size[0]:
                        for xn in range(x+1, x+11):
                            if pixdata[xn, y] != (0,0,0) and pixdata[xn, y] != (255, 0, 0):
                                pixdata[xn, y] = (255, 210, 0)
                    if x-10>=0:
                        for xn in range(x-10, x):
                            if pixdata[xn, y] != (0,0,0) and pixdata[xn, y] != (255, 0, 0):
                                pixdata[xn, y] = (255, 210, 0)
                    if y+10<im.size[1]:
                        for yn in range(y+1, y+11):
                            if pixdata[x, yn] != (0,0,0) and pixdata[x, yn] != (255, 0, 0):
                                pixdata[x, yn] = (255, 210, 0)
                    if y-10>=0:
                        for yn in range(y-10, y):
                            if pixdata[x, yn] != (0,0,0) and pixdata[x, yn] != (255, 0, 0):
                                pixdata[x, yn] = (255, 210, 0)
                        
                    pixdata[x, y] = (255, 0, 0)
                    

        im.save("red_coloured.png")
        im = Image.open("red_coloured.png")
        im = im.convert('RGB')
        pixdata = im.load()
        print(im.size)
        #print(pixdata[x,y])
        #for pixel in im.getdata():

        for y in range(im.size[1]):
            for x in range(im.size[0]):
                if pixdata[x, y] != (255, 0, 0) and pixdata[x, y] != (255, 210, 0):
                    pixdata[x, y] = (0, 255, 0)
                # if x>=250 and x<=330 and y<=530 and y>=450:
                #      pixdata[x, y] = (0, 0, 255)
        initial_x = 9.49
        initial_y = 76.33
        increment_x = 0.003
        increment_y = 0.0001
        
        im.crop((270, 450, 330, 530)).save('area.jpg')

                #Alapuzha = 215, 375    9.49, 76.33
                #Ambalapuzha = 255 , 505
                #0.12, 0.02


            
                    

        im.save("./static/images/final.png")
        background = Image.open("./static/images/before.jpg").convert('RGBA')
        foreground = Image.open("./static/images/final.png").convert('RGBA')
        foreground.putalpha(96)

        background.paste(foreground, (0,0) , foreground)
        background.save("./static/images/final3.png")
        
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img=mpimg.imread('./static/images/final3.png')
        before=mpimg.imread('./static/images/before.jpg')
        after=mpimg.imread('./static/images/after.jpg')
        fig = plt.figure(figsize=(21, 7))
        ax1 = fig.add_subplot(1, 3, 1)
        plt.imshow(before)
        ax2 = fig.add_subplot(1, 3, 2)
        plt.imshow(after)
        ax3 = fig.add_subplot(1, 3, 3)
        plt.imshow(img)
        ax1.title.set_text('Before')
        ax2.title.set_text('After')
        ax3.title.set_text('HeatMap')
        ax1.set_anchor('N')
        ax2.set_anchor('N')
        ax3.set_anchor('N')
        plt.show()
        return background   


#NEXT CODE
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import os
import numpy as np
from PIL import Image
import sys



def findExistingCenters(x, y, im):
    pixdata = im.load()
    for i in range(x-14, x+14):
        for j in range(y-44, y+45):
            if i>=0 and j>=0 and j<im.size[1] and i<im.size[0]:
                if pixdata[i, j] == (0, 0, 255):
                    return True
    return False
def checkValidityOfCenter(x_value, y_value, x, y, im, centers):
    pixdata = im.load()
    for i in range(x-7, x+8):
        for j in  range(y-22, y+23):
            
            if i>=0 and j>=0 and j<im.size[1] and i<im.size[0]:
                
                if pixdata[i, j][0]>200 and pixdata[i, j][1]<130 and pixdata[i, j][2]<130:
                # if pixdata[i, j][1]>200 and  pixdata[i, j][0]<130 and pixdata[i, j][2]<130:
                
                    pixdata[x, y] = (0, 0, 255)
                    centers.append([x_value, y_value])
                    return
def getCenters():
    file = 'area.jpg'
    # imageA = cv2.imread(file)
    initial_x = 9.655
    initial_y = 76.3375
    increment_x = 0.003
    increment_y = 0.0001
    im = Image.open(file)
    print(im.size)
    pixdata = im.load()
    centers = []
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            x_value = initial_x + x * increment_x
            y_value = initial_y + y * increment_y
            if pixdata[x, y][1]>200 and  pixdata[x, y][0]<130 and pixdata[x, y][2]<130:
            # if pixdata[x,y][0]>200 and pixdata[x,y][1]<130 and pixdata[x,y][2]<130:
            
                # print(pixdata[x, y])
                if not findExistingCenters(x, y, im):
                    checkValidityOfCenter(x_value, y_value, x, y, im, centers)

    print(centers)
    im.save("centers.png") 
    # im.show()
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img=mpimg.imread('centers.png')
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    plt.imshow(img)
    ax1.title.set_text('Health Camp Centers')
    
    ax1.set_anchor('N')
    
    plt.show()
    
    return centers        



from imageai.Detection import ObjectDetection
import os
import datetime
import pandas as pd
# import ibm_db
execution_path = os.getcwd()
# conn = ibm_db.connect("DATABASE=BLUDB;HOSTNAME=dashdb-txn-sbox-yp-lon02-01.services.eu-gb.bluemix.net;PORT=50000;PROTOCOL=TCPIP;UID=fkk32348;PWD=kzpx8xvfvg-4642k","fkk32348","kzpx8xvfvg-4642k")
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

final_columns = ['Timestamp', 'Latitude', 'Longitude', 'People']
lat_lon = [[9.655, 76.3375], [9.780999999999999, 76.3375], [9.7, 76.33840000000001], [9.825999999999999, 76.34020000000001], [9.745, 76.3422], [9.673, 76.3429]]
execution_path = execution_path + "/images2/"

from PIL import Image

def crop(path, input):
    im = Image.open(input)
    imgwidth, imgheight = im.size
    height = int(imgheight/2)
    width = int(imgwidth/2)
    k = 0
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            a.save(os.path.join(path,"IMG-%s.png" % k))
            k +=1

def image_processing():
        counter = 0
        df = pd.DataFrame()
        for file in os.listdir(execution_path):
            if file.endswith(".jpg"):
                    print(file)
                    filepath = execution_path + file
                    crop(execution_path , execution_path + file )
                    count = 0
                    for i in range(0,4):

                            detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "IMG-%s.png" % i), output_image_path=os.path.join(execution_path , "IMG-%s.png" % i ))
                            for eachObject in detections:
                                    if eachObject["name"]=="person":
                                            count = count + 1
                            print(count)
                    df = df.append( pd.Series([datetime.datetime.now(), lat_lon[counter][0], lat_lon[counter][1], count], index=final_columns), ignore_index=True)
                    # stmt = ibm_db.exec_immediate(conn, "INSERT INTO SURVEY VALUES(%s, %s, %s, %s);", datetime.datetime.now(), lat_lon[counter][0], lat_lon[counter][1], count)
                    # print("Number of affected rows: ", ibm_db.num_rows(stmt))
                    counter = counter + 1
                    print(count)
                    print(df)

        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import matplotlib.gridspec as gridspec
        img=mpimg.imread(filepath)
        img_0=mpimg.imread(execution_path + "IMG-0.png")
        img_1=mpimg.imread(execution_path + "IMG-1.png")
        img_2=mpimg.imread(execution_path + "IMG-2.png")
        img_3=mpimg.imread(execution_path + "IMG-3.png")
        fig = plt.figure(figsize=(10, 10))

        ax1 = fig.add_subplot(1, 1, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.show()

        fig2 = plt.figure(figsize=(10, 10))
        plt.title('People Detection with Bounding Boxes')
        ax2 = fig2.add_subplot(2, 2, 1)
        plt.imshow(img_0)
        ax3 = fig2.add_subplot(2,2, 2)
        plt.imshow(img_1)
        ax4 = fig2.add_subplot(2, 2, 3)
        plt.imshow(img_2)
        ax5 = fig2.add_subplot(2, 2, 4)
        plt.imshow(img_3)
        
        plt.show()
        

        # gs1 = gridspec.GridSpec(4, 2)
        # gs1.update(left=0.05, right=0.48, wspace=0.05)
        # ax1 = plt.subplot(gs1[:-2, :])
        # plt.imshow(img)
        # ax2 = plt.subplot(gs1[-2, :-1])
        # plt.imshow(img_0)
        # ax3 = plt.subplot(gs1[-2, -1])
        # plt.imshow(img_1)
        # ax4 = plt.subplot(gs1[-1, :-1])
        # plt.imshow(img_2)
        # ax4 = plt.subplot(gs1[-1, -1])
        # plt.imshow(img_3)
        # plt.show()
        # gs1 = gridspec.GridSpec(16, 8)
        # gs1.update(left=0.05, right=0.48, wspace=0.05)
        # ax1 = plt.subplot(gs1[:-8, :])
        # plt.imshow(img)
        # ax2 = plt.subplot(gs1[-8, :-4])
        # plt.imshow(img_0)
        # ax3 = plt.subplot(gs1[-8, -4])
        # plt.imshow(img_1)
        # ax4 = plt.subplot(gs1[-8, :-4])
        # plt.imshow(img_2)
        # ax4 = plt.subplot(gs1[-8, -4])
        # plt.imshow(img_3)
        # plt.title('Detection of People')
        # plt.show()
        


        # ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2, rowspan = 2)
        # plt.imshow(img)
        # ax2 = plt.subplot2grid((4, 2), (2, 0), colspan=1, rowspan = 1)
        # plt.imshow(img_0)
        # ax3 = plt.subplot2grid((4, 2), (2, 1), colspan=1, rowspan = 1)
        # plt.imshow(img_1)
        # ax4 = plt.subplot2grid((4, 2), (3, 0), colspan=1, rowspan = 1)
        # plt.imshow(img_2)
        # ax5 = plt.subplot2grid((4, 2), (3, 1), colspan=1, rowspan = 1)
        # plt.imshow(img_3)
        # ax1 = fig.add_subplot(1, 2, 1)
        
        # fig2 = fig.add_subplot(1, 2, 2)
        # ax2 = fig2.add_subplot(2, 2, 1)
        
        # ax3 = fig2.add_subplot(2, 2, 2)
        
        # ax4 = fig2.add_subplot(2, 2, 3)
       
        # ax5= fig2.add_subplot(2, 2, 4)
        
        # # ax1.title.set_text('Before')
        # # ax2.title.set_text('After')
        # # ax3.title.set_text('HeatMap')
        # ax1.set_anchor('N')
        # ax2.set_anchor('N')
        # ax3.set_anchor('N')
        # plt.show()
        

        
            


        df.columns = final_columns
        df.to_csv('dataset.csv', header=True, index=False)

import matplotlib.pyplot as plt
import time
import random
from threading import Thread
import math

    
def func1():
    xdata = []
    ydata = []
    line, = axes.plot(xdata, ydata, 'r-')
    radius = 5
    center = [0,0]
    for a in range(0, 360, 30):
        angle = a * math.pi / 180
        for r in range(0,radius+1):
            x = center[0] + (r * math.cos(angle))
            y = center[1] + (r * math.sin(angle))            
            xdata.append(x)
            ydata.append(y)
            line.set_xdata(xdata)
            line.set_ydata(ydata)
            plt.draw()
            plt.pause(1e-17)
            time.sleep(0.1)
        angle = (a+15) * math.pi / 180
        for r in range(radius,-1,-1):
            x = center[0] + (r * math.cos(angle))
            y = center[1] + (r * math.sin(angle))
            xdata.append(x)
            ydata.append(y)
            line.set_xdata(xdata)
            line.set_ydata(ydata)
            plt.draw()
            plt.pause(1e-17)
    plt.show(block=False)
    plt.close()
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
import numpy as np

def function():
    df = pd.DataFrame({
        'x': [9.655,9.781,9.7,9.826,9.745,9.673],
        'y': [76.3375,76.3375,76.3384,76.3402,76.3422,76.3429]
    })

    # X = [[23,33],[2,4],[12,39],[20,36],[28,30],[69,7],[4,5],[33,55],[60,22]]
    # df = pd.Dataframe(X)
    # print(df)
    colmap = {1: 'r', 2: 'g', 3: 'b'}
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(df)
    labels = kmeans.predict(df)
    df['labels']=labels
    centroids = kmeans.cluster_centers_
    fig = plt.figure(figsize=(5, 5))


    # colors = map(lambda x: colmap[x+1], labels)
    # for key in colors.keys():
    #     print(key)
    #     print(colors[key])
    colors = []
    for value in labels:
        colors.append(colmap[value+1])
    df['color']=colors
    print(df)
    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for idx, centroid in enumerate(centroids):
        # plt.scatter(*centroid, color=colmap[idx+1])
        plt.scatter(*centroid, color='blue')
    plt.xlim(9.6, 9.9)
    plt.ylim(76, 77)
    plt.title('Clustering to identify places to send ambulance')
    plt.show()
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pandas as pd
import geopy.distance
def supply_drone():
    origin = [9.808, 76.33760000000001]
    arr = [[9.655, 76.3375], [9.780999999999999, 76.3375], [9.7, 76.33840000000001], [9.825999999999999, 76.34020000000001], [9.745, 76.3422], [9.673, 76.3429]]

    dist = {}
    count = 0
    for i in arr:
        distance = geopy.distance.vincenty(origin, i).km
        dist[count] = distance
        count = count + 1
        print (distance)
    import operator
    sorted_x = sorted(dist.items(), key=operator.itemgetter(1))
    print (sorted_x)
    sorted_list = []
    for tuple in sorted_x:
        sorted_list.append(arr[tuple[0]])
    print(sorted_list)

    df = pd.DataFrame(sorted_list)
    df.columns = ['x', 'y']
    fig = plt.figure()
    
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('Supply Drone supplying packages to people')
    ax1.set_ylim(76.33, 76.35)
    ax1.set_xlim(9.6, 9.9)
    #df = pd.DataFrame({'x': [9.655,9.781,9.7,9.826,9.745,9.673],'y': [76.3375,76.3375,76.3384,76.3402,76.3422,76.3429]})

    ax1.scatter(df['x'],df['y'], color = 'blue')
    ax1.scatter(9.808, 76.33760000000001, color='red')
    in_x = 9.808
    in_y = 76.33760000000001
    def animate(i):
        x = df.iloc[i%6]['x']
        y = df.iloc[i%6]['y']
        plt.plot([in_x,x],[in_y,y],'k-')
        

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()

import cv2
import numpy as np
from skimage.morphology import skeletonize,thin
from skimage.util import invert
import os


def road_detection(fname):
    fname = './static/images/flooded2.jpg'
    print(os.path.exists(fname))
    # im1 = cv2.cv.LoadImage(fname, CV_LOAD_IMAGE_COLOR)
    im1 = cv2.imread(fname, 0)
    print(im1.size)

    # cv2.imshow('',im1)
    #cv2.waitKey(0)

    '''
    im = cv2.Canny(im,200,400)


    cv2.imshow('',im)
    cv2.waitKey(0)



    im = cv2.medianBlur(im, 5)

    cv2.imshow('',im)
    cv2.waitKey(0)
    '''

    #im = cv2.threshold(im, 0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    im = cv2.threshold(im1, 130 , 255, cv2.THRESH_BINARY)


    im=im[1]

    #cv2.imshow('',im)
    #cv2.waitKey(0)

    #im = cv2.medianBlur(im, 5)

    #cv2.imshow('',im)
    #cv2.waitKey(0)


    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im)


    sizes = stats[1:, -1]; nb_components = nb_components - 1

    min_size = 1000

    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size and sizes[i]<2000:
            img2[output == i + 1] = 1
            #cv2.imshow('',img2)
            #cv2.waitKey(0)
            print(sizes[i])
            #img2 = np.zeros((output.shape))
            

    #cv2.imshow('',img2)
    #cv2.waitKey(0)

    skeleton = skeletonize(img2)

    #skeleton = thin(img2,max_iter=25)



    skeleton = np.array(skeleton, dtype=np.uint8)

    print(skeleton)

    for i in range(len(skeleton)):
        for j in range(len(skeleton[0])):
            if skeleton[i][j]==1:
                skeleton[i][j]=255

    print(skeleton)

    #cv2.imshow('',skeleton)
    #cv2.waitKey(0)


    #img2=np.array(img2, dtype=np.uint8)
    image = cv2.Canny(skeleton,100,200)
    '''
    for i in range(len(im)):
        for j in range(len(im)):
            if im[i][j]==1:
                im[i][j]=255
       
    '''
    #cv2.imshow('',image)
    #cv2.waitKey(0)


    #new_img = Image.blend(im, image, 0.5)
    dst = cv2.addWeighted(im1,0.4,image,0.8,0)
    # cv2.imshow('Road Detection',dst)
    # cv2.waitKey(0)

    cv2.imwrite(r"./static/images/road2.jpg",dst)
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img=mpimg.imread('./static/images/flooded2.jpg')
    img2=mpimg.imread('./static/images/road2.jpg')
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.title.set_text('Before')
    plt.imshow(img)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.title.set_text('After Road Detection')
    plt.imshow(img2)
    plt.show()
    
if __name__ == '__main__':
    imageCompare('./static/images/before.jpg', './static/images/after.jpg')
    input("Press Enter to continue...")
    centers = getCenters()
    input("Press Enter to continue...")
    plt.show()
    axes = plt.gca()
    axes.set_title('Survey Drone Path From HealthCamp')
    N=15
    Nx=N/2 
    axes.set_xlim(-5, 5)
    axes.set_ylim(-5, 5)
    func1()
    input("Press Enter to continue...")
    image_processing()
    input("Press Enter to continue...")
    supply_drone()
    input("Press Enter to continue...")
    function()
    input("Press Enter to continue...")
    road_detection('./static/images/flooded2.jpg')
    