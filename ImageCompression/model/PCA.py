import numpy as np


#数据中心化
def Z_centered(dataMat):
	rows,cols=dataMat.shape
	meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
	meanVal = np.tile(meanVal,(rows,1))
	newdata = dataMat-meanVal
	return newdata, meanVal
 
#协方差矩阵
def Cov(dataMat):
	rows = dataMat.shape[0]
	meanVal = np.mean(dataMat,0) #压缩行，返回1*cols矩阵，对各列求均值
	meanVal = np.tile(meanVal, (rows,1)) #返回rows行的均值矩阵
	Z = dataMat - meanVal
	Zcov = (1/(rows-1))*Z.T * Z
	return Zcov
	
#得到最大的k个特征值和特征向量
def EigDV(covMat, k):
	D, V = np.linalg.eig(covMat) # 得到特征值和特征向量
	eigenvalue = np.argsort(D)
	K_eigenValue = eigenvalue[-1:-(k+1):-1]
	K_eigenVector = V[:,K_eigenValue]
	return K_eigenValue, K_eigenVector
	
#得到降维后的数据
def getlowDataMat(DataMat, K_eigenVector):
	return DataMat * K_eigenVector
 
#重构数据
def Reconstruction(lowDataMat, K_eigenVector, meanVal):
	reconDataMat = lowDataMat * K_eigenVector.T + meanVal
	return reconDataMat
 
#PCA算法
def PCA(data, k):
	dataMat = np.float32(np.asmatrix(data))
	#数据中心化
	dataMat, meanVal = Z_centered(dataMat)
	#计算协方差矩阵
	covMat = np.cov(dataMat, rowvar=0)
	#得到最大的k个特征值和特征向量
	_, V = EigDV(covMat, k)
	#得到降维后的数据
	lowDataMat = getlowDataMat(dataMat, V)
	#重构数据
	reconDataMat = Reconstruction(lowDataMat, V, meanVal)
	return lowDataMat, reconDataMat