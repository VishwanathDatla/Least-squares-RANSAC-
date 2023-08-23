from re import U
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


x1 = 5
y1 = 5
xp1 = 100
yp1 = 100
x2 = 150
y2 = 5
xp2 = 200
yp2 = 80
x3 = 150
y3 = 150
xp3 = 220
yp3 = 80
x4 = 5
y4 = 150
xp4 = 100
yp4 = 200

row1 = [-x1,-y1,-1,0,0,0,x1 *xp1,y1 * xp1,xp1]
row2 = [0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1]
row3 = [-x2,-y2,-1,0,0,0,x2 *xp2,y2 * xp2,xp2]
row4 = [0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2]
row5 = [-x3,-y3,-1,0,0,0,x3*xp3,y3 * xp3,xp3]
row6 = [0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3]
row7 = [-x4,-y4,-1,0,0,0,x4*xp4,y4 * xp4,xp4]
row8 = [0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4]

mat_a = np.row_stack((row1,row2,row3,row4,row5,row6,row7,row8))

mat_a1 = np.matmul(mat_a,mat_a.T)

e_val,e_vec = np.linalg.eig(mat_a1)
e_val = np.abs(e_val)

index = e_val.argsort()[::-1]
e_val = e_val[index]

index = e_val.argsort()[::-1]  
U = e_vec[:,index]
e_vec = U

print(U)

sval = []
for i in range(0,len(e_val)):
    if e_val[i]!=0:
        sval.append(np.sqrt(np.abs(e_val[i])))

sval = np.sort(sval)
leftsingularvalues = np.flipud(sval)

#print(sval)

D=np.diag(sval)
temp = np.array([[0]]*8)
D = np.append(D, temp, axis = 1)
#print(D)

mat_a2 = np.matmul(mat_a.T,mat_a)

e_val1,e_vec1 = np.linalg.eig(mat_a2)
e_val1 = np.abs(e_val1)
righteigenvalues = e_val1[e_val1.argsort()[::-1]]

index1 = e_val1.argsort()[::-1]  
V = e_vec1[:,index1]

print(V)

H = V[:,8]
H = np.reshape(H,(3,3))
print(H)