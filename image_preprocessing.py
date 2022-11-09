'''
航拍图片 training samples 预处理

Selina Wang
Feb. 2022

'''
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from PIL import Image 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from glob import glob
import os



def main():
	isExist = os.path.exists("processed_images")
	if not isExist:
		os.mkdir("processed_images")

	
	folder = input('输入原图文件夹名称：')
	k = float(input('输入图片缩小比例：')) # porportion of reduced image
	
	# generate base means from a base image
	base_means = np.array([153, 141, 125])
	
	image_names = glob("./{}/*.JPG".format(folder))
	
	for filename in image_names:
		image = Image.open(filename)
		name = filename.split("/")[2]
		image_processed = process_im(image, k, base_means)
		image_processed.save('./processed_images/{}'.format(name))



def get_array(image):
	image_array = np.array(image)
	image_array = image_array.astype("float32")
	return image_array, image.size

def reduce_im(image, k):
	w,h = image.size
	w = int(k*w)
	h = int(k*h)
	image = image.resize((w,h))
	
	return image

# adjusts each channel of image based on base image
# adjusted image will have the same mean for each channel as the base image
def center_im(image_array,base_means):

	means = image_array.mean(axis=(0,1), dtype='float64')

	# adjust according to base image
	adjust = np.diag(base_means/means)
	
	# center:
	image_array = image_array@adjust

	# confirm it had the desired effect
	means = image_array.mean(axis=(0,1), dtype='float64')
	#print('Means: %s' % means)
	#print('Mins: %s, Maxs: %s' % (image_array.min(axis=(0,1)), image_array.max(axis=(0,1))))
	
	return image_array

def process_im(image, k, base_means):
	image = reduce_im(image, k)
	image_array, (w, h) = get_array(image)
	image_array = center_im(image_array, base_means)

	image_processed = Image.fromarray(image_array.astype(np.uint8))
	
	return image_processed


'''
# takes an image, returns the means of each RBG channel
def get_base_means(image, k):

	image = reduce_im(image, k)
	image_array = np.array(image)
	image_array = image_array.astype('float32')


	base_means = image_array.mean(axis=(0,1), dtype='float64')

	return base_means
'''

if __name__=="__main__":
	main()