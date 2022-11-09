'''
Classification 模型训练

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
import pickle


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def main():
	# label data
	treeTraining = get_training_data(1,'tree')
	soilTraining = get_training_data(0,'soil')

	# combine data into one df
	allTraining = treeTraining.append(soilTraining)
	allTraining = allTraining.round(decimals = 0)
	allTraining = allTraining.astype(int)

	# remove duplicated pixels
	allTraining.drop_duplicates(inplace = True)

	# split data further into training and testing set
	X_train, X_test, y_train, y_test = train_test_split(allTraining.drop(columns=['label']), allTraining['label'], test_size=0.3)

	
	# hyperparameter tuning
	print("tuning parameters")
	param_grid = {'C': [0.1,1,10], 'gamma': [1,0.1,0.01]}
	grid = GridSearchCV(svm.SVC(kernel='linear'),param_grid,refit=True,verbose=2)
	grid.fit(X_train,y_train)


	# train model with best parameters
	print("best parameters:", grid.best_params_)
	clf_svm = svm.SVC(kernel='linear',C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
	clf_svm.fit(X_train, y_train)
	y_pred = clf_svm.predict(X_test)


	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
	print("Confusion Matrix:",metrics.confusion_matrix(y_test, y_pred))

	filename = 'classification_model.sav'
	pickle.dump(clf_svm, open(filename, 'wb'))

	# test if it works:
	'''
	image = Image.open("./DJI_0199.JPG")
	image_array,(w,h) = get_array(image)
	image.close()
	processed_image = flatten_im(image_array)

	final_model = pickle.load(open(filename, 'rb'))
	test_results = final_model.predict(processed_image)

	classified_image = test_results.reshape(300, 400)
	plt.imshow(classified_image)
	'''


def get_training_data(label, category):
    sampleTraining = pd.DataFrame()
    samples = glob('./training/{}/*'.format(category))

    for sample in samples:
        image = Image.open(sample)
        image_array = np.array(image)        
        image_array = image_array.reshape((image_array.shape[0]*image_array.shape[1]), image_array.shape[2])
        
        #print(image_array.shape)
        image_df = pd.DataFrame(image_array,columns = ['r','g','b','extra'])
        sampleTraining = sampleTraining.append(image_df)

    sampleTraining.drop(columns = ['extra'], inplace = True)
    sampleTraining['label'] = label

    return sampleTraining

def get_array(image):
	image_array = np.array(image)
	image_array = image_array.astype("float32")
	return image_array, image.size


def flatten_im(image_array):
	image_flat = image_array.reshape((image_array.shape[0] * image_array.shape[1]), image_array.shape[2])
	return image_flat


if __name__=="__main__":
	main()