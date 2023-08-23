import pandas as pd
import numpy as np 
import random
import matplotlib.pyplot as plt
df = pd.read_csv('ENPM673_hw1_linear_regression_dataset - Sheet1.csv')
x_data = df.iloc[:,0]
y_data = df.iloc[:,-1]
covariance_mat = np.zeros((2,2))
def variance(x_data,y_data):
    deviations = [(x - np.average(x_data)) * (y - np.average(y_data)) for x,y in zip(x_data, y_data)]
    variance = sum(deviations)/len(x_data)
    return variance 
covariance_mat[0][0] = variance(x_data, x_data)
covariance_mat[0][1] = variance(x_data, y_data)
covariance_mat[1][0] = variance(y_data, x_data)
covariance_mat[1][1] = variance(y_data, y_data)

e_val,e_vec = np.linalg.eig(covariance_mat)
origin = [np.average(x_data),np.average(y_data)]

'''
#CODE TO PLOT DATA WITH EIGEN VECTORS 

plt.scatter(x_data,y_data)
plt.quiver(*origin, *e_vec[:,0], color=['r'], scale=21)
plt.quiver(*origin, *e_vec[:,1], color=['b'], scale=21)
plt.show()
'''

def mean(x):
    sum = 0
    for i in range(0,len(x)):
        sum = sum + x[i]
    return sum/len(x)


def normalize(x):
    x_min = np.min(x)
    x_max = np.max(x)
    return [((a - x_min) / (x_max - x_min))for a in x],max


x_data,maxx = normalize(x_data)
y_data,maxy = normalize(y_data)
x = np.row_stack(x_data)
y = np.row_stack(y_data)

def moment_matrix(x,xbar):
        u = []
        for i in range(0,len(x)):
            u.append(x[i] - xbar)
        return u

def second_moment_matrix(U) :
        return np.matmul(U.T,U)
        
'''
#CODE TO PLOT DATA WITH TOTAL LEAST SQUARES 
xbar = mean(x)
ybar = mean(y)
ux = moment_matrix(x,xbar)
uy = moment_matrix(y,ybar)

U = np.column_stack((ux,uy))
second_moment = second_moment_matrix(U)
eigenvalues,eigenvectors = np.linalg.eig(second_moment)
N = eigenvectors[:,1]
dist = N[0]*xbar + N[1]*ybar
val = (dist - (N[0]*x))/N[1]
plt.plot(x, y,'o')
plt.plot(x,val)
plt.show()
'''


def ransac(x_data, y_data, threshold):
    iterations = np.max((np.log(0.1) / np.log(1 - np.power((1 - 0.2), 2)),100))
    while iterations>0:
        outliers=0
        max = len(x_data)
        points = np.column_stack((x_data, y_data))
        points = np.ndarray.tolist(points)
        p1,p2 = random.sample(points,2)
        p1 = np.array(p1)
        p2 = np.array(p2)
        for p in points:
           p = np.array(p)
           if np.all(p==p2):
               continue
           d= (np.cross(p2-p1, p-p1))/(np.linalg.norm(p2-p))
           if d>threshold:
               outliers+=1
        if outliers<max:
            max = outliers
            p_final1 = p1
            p_final2 = p2
        iterations = iterations-1 
    return p_final1, p_final2

#CODE TO PLOT DATA USING RANSAC
p1,p2 = ransac(x_data,y_data,0.5)
p1 = p1.tolist()
p2 = p2.tolist()
x_values = [p1[0], p2[0]]
y_values = [p1[1], p2[1]]
plt.scatter(x_data,y_data)
plt.plot(x_values,y_values)
plt.show()


'''
#CODE TO PLOT DATA WITH LINEAR LEAST SQUARES FITTING
w = linear_LS(x_data,y_data)
x= np.linspace(15,90,20)
y = w[1]*x +w[0]
plt.scatter(x_data,y_data)
plt.plot(x,y, 'r')
plt.show()        
'''    
    
