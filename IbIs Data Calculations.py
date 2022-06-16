%matplotlib inline
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from numpy import diff
from sympy import diff, sin, exp
plt.style.use('seaborn-whitegrid')
import pandas as pd
x1 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Ib(uA)-1pA.txt")
x2 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Ib(uA)-1pB.txt")
x3 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Ib(uA)-1pC.txt")
x4 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Ib(uA)-1pD.txt")
x5 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Ib(uA)-1pE.txt")
x6 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Ib(uA)-2pA.txt")
x7 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Ib(uA)-2pB.txt")
x8 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Ib(uA)-2pC.txt")
x9 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Ib(uA)-2pD.txt")
x10 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Ib(uA)-2pE.txt")
x11 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Ib(uA)-2pF.txt")
m_1 = np.array(x1, dtype=np.float64) #array that includes all Ib values for channel 1pA
m_2 = np.array(x2, dtype=np.float64)
m_3 = np.array(x3, dtype=np.float64)
m_4 = np.array(x4, dtype=np.float64)
m_5 = np.array(x5, dtype=np.float64)
m_6 = np.array(x6, dtype=np.float64)
m_7 = np.array(x7, dtype=np.float64)
m_8 = np.array(x8, dtype=np.float64)
m_9 = np.array(x9, dtype=np.float64)
m_10 = np.array(x10, dtype=np.float64)
m_11 = np.array(x11, dtype=np.float64)


m1A=len(m_1) - 50
m2A=len(m_1) - 40
m3A=len(m_1) - 30
m4A=len(m_1) - 20
m5A=len(m_1) - 10
m6A=len(m_1) - 1

m1B=len(m_2) - 50
m2B=len(m_2) - 40
m3B=len(m_2) - 30
m4B=len(m_2) - 20
m5B=len(m_2) - 10
m6B=len(m_2) - 1

m1C=len(m_3) - 50
m2C=len(m_3) - 40
m3C=len(m_3) - 30
m4C=len(m_3) - 20
m5C=len(m_3) - 10
m6C=len(m_3) - 1

m1D=len(m_4) - 50
m2D=len(m_4) - 40
m3D=len(m_4) - 30
m4D=len(m_4) - 20
m5D=len(m_4) - 10
m6D=len(m_4) - 1

m1E=len(m_5) - 50
m2E=len(m_5) - 40
m3E=len(m_5) - 30
m4E=len(m_5) - 20
m5E=len(m_5) - 10
m6E=len(m_5) - 1

m1A_2=len(m_6) - 50
m2A_2=len(m_6) - 40
m3A_2=len(m_6) - 30
m4A_2=len(m_6) - 20
m5A_2=len(m_6) - 10
m6A_2=len(m_6) - 1

m1B_2=len(m_7) - 50
m2B_2=len(m_7) - 40
m3B_2=len(m_7) - 30
m4B_2=len(m_7) - 20
m5B_2=len(m_7) - 10
m6B_2=len(m_7) - 1

m1C_2=len(m_8) - 50
m2C_2=len(m_8) - 40
m3C_2=len(m_8) - 30
m4C_2=len(m_8) - 20
m5C_2=len(m_8) - 10
m6C_2=len(m_8) - 1

m1D_2=len(m_9) - 50
m2D_2=len(m_9) - 40
m3D_2=len(m_9) - 30
m4D_2=len(m_9) - 20
m5D_2=len(m_9) - 10
m6D_2=len(m_9) - 1

m1E_2=len(m_10) - 50
m2E_2=len(m_10) - 40
m3E_2=len(m_10) - 30
m4E_2=len(m_10) - 20
m5E_2=len(m_10) - 10
m6E_2=len(m_10) - 1

m1F_2=len(m_11) - 50
m2F_2=len(m_11) - 40
m3F_2=len(m_11) - 30
m4F_2=len(m_11) - 20
m5F_2=len(m_11) - 10
m6F_2=len(m_11) - 1


y1 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Is(uA)-1pA.txt")
y2 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Is(uA)-1pB.txt")
y3 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Is(uA)-1pC.txt")
y4 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Is(uA)-1pD.txt")
y5 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Is(uA)-1pE.txt")
y6 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Is(uA)-2pA.txt")
y7 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Is(uA)-2pB.txt")
y8 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Is(uA)-2pC.txt")
y9 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Is(uA)-2pD.txt")
y10 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Is(uA)-2pE.txt")
y11 = pd.read_csv("/Users/giulia/Documents/IbIs Data/Is(uA)-2pF.txt")

n_1 = np.array(y1, dtype=np.float64)#array that includes all Is values for channel 1pA
n_2 = np.array(y2, dtype=np.float64)
n_3 = np.array(y3, dtype=np.float64)
n_4 = np.array(y4, dtype=np.float64)
n_5 = np.array(y5, dtype=np.float64)
n_6 = np.array(y6, dtype=np.float64)
n_7 = np.array(y7, dtype=np.float64)
n_8 = np.array(y8, dtype=np.float64)
n_9 = np.array(y9, dtype=np.float64)
n_10 = np.array(y10, dtype=np.float64)
n_11 = np.array(y11, dtype=np.float64)




n_1_new = np.array([]) 
m_1_new = np.array([])
for i in range(len(n_1)-1):
    n_1_new = np.append(n_1_new, n_1[i])
    m_1_new = np.append(m_1_new, m_1[i])
    for i in range(len(n_1_new)-1):
        j = i+1
        if n_1_new[i+1]-n_1_new[i]>5: 
            n_1_new = n_1_new[j:]
            m_1_new = m_1_new[j:]


        
n_2_new = np.array([]) 
m_2_new = np.array([])
for i in range(len(n_2)-1):
    n_2_new = np.append(n_2_new, n_2[i])
    m_2_new = np.append(m_2_new, m_2[i])
    for i in range(len(n_2_new)-1):
        j = i+1
        if n_2_new[i+1]-n_2_new[i]>5: 
            n_2_new = n_2_new[j:]
            m_2_new = m_2_new[j:]


        
n_3_new = np.array([]) 
m_3_new = np.array([])
for i in range(len(n_3)-1):
    n_3_new = np.append(n_3_new, n_3[i])
    m_3_new = np.append(m_3_new, m_3[i])
    for i in range(len(n_3_new)-1):
        j = i+1
        if n_3_new[i+1]-n_3_new[i]>5: 
            n_3_new = n_3_new[j:]
            m_3_new = m_3_new[j:] 
        
n_4_new = np.array([]) 
m_4_new = np.array([])
for i in range(len(n_4)-1):
    n_4_new = np.append(n_4_new, n_4[i])
    m_4_new = np.append(m_4_new, m_4[i])
    for i in range(len(n_4_new)-1):
        j = i+1
        if n_4_new[i+1]-n_4_new[i]>5: 
            n_4_new = n_4_new[j:]
            m_4_new = m_4_new[j:]
 
        
        
n_5_new = np.array([]) 
m_5_new = np.array([])
for i in range(len(n_5)-1):
    n_5_new = np.append(n_5_new, n_5[i])
    m_5_new = np.append(m_5_new, m_5[i])
    for i in range(len(n_5_new)-1):
        j = i+1
        if n_5_new[i+1]-n_5_new[i]>5: 
            n_5_new = n_5_new[j:]
            m_5_new = m_5_new[j:]

 
n_6_new = np.array([]) 
m_6_new = np.array([])
for i in range(len(n_6)-1):
    n_6_new = np.append(n_6_new, n_6[i])
    m_6_new = np.append(m_6_new, m_6[i])
    for i in range(len(n_6_new)-1):
        j = i+1
        if n_6_new[i+1]-n_6_new[i]>5: 
            n_6_new = n_6_new[j:]
            m_6_new = m_6_new[j:]


n_7_new = np.array([]) 
m_7_new = np.array([])
for i in range(len(n_7)-1):
    n_7_new = np.append(n_7_new, n_7[i])
    m_7_new = np.append(m_7_new, m_7[i])
    for i in range(len(n_7_new)-1):
        j = i+1
        if n_7_new[i+1]-n_7_new[i]>5: 
            n_7_new = n_7_new[j:]
            m_7_new = m_7_new[j:]


        
        
n_8_new = np.array([]) 
m_8_new = np.array([])
for i in range(len(n_8)-1):
    n_8_new = np.append(n_8_new, n_8[i])
    m_8_new = np.append(m_8_new, m_8[i])
    for i in range(len(n_8_new)-1):
        j = i+1
        if n_8_new[i+1]-n_8_new[i]>5: 
            n_8_new = n_8_new[j:]
            m_8_new = m_8_new[j:]
        
n_9_new = np.array([]) 
m_9_new = np.array([])
for i in range(len(n_9)-1):
    n_9_new = np.append(n_9_new, n_9[i])
    m_9_new = np.append(m_9_new, m_9[i])
    for i in range(len(n_9_new)-1):
        j = i+1
        if n_9_new[i+1]-n_9_new[i]>5: 
            n_9_new = n_9_new[j:]
            m_9_new = m_9_new[j:]

        
n_10_new = np.array([]) 
m_10_new = np.array([])
for i in range(len(n_10)-1):
    n_10_new = np.append(n_10_new, n_10[i])
    m_10_new = np.append(m_10_new, m_10[i])
    for i in range(len(n_10_new)-1):
        j = i+1
        if n_10_new[i+1]-n_10_new[i]>5: 
            n_10_new = n_10_new[j:]
            m_10_new = m_10_new[j:]


n_11_new = np.array([]) 
m_11_new = np.array([])
for i in range(len(n_11)-1):
    n_11_new = np.append(n_11_new, n_11[i])
    m_11_new = np.append(m_11_new, m_11[i])
    for i in range(len(n_11_new)-1):
        j = i+1
        if n_11_new[i+1]-n_11_new[i]>5: 
            n_11_new = n_11_new[j:]
            m_11_new = m_11_new[j:]
 
        
        
#d_Ib = np.array([])
#d_Is = np.array([])
#dPdI = np.array([])
#for i in range(len(n)):
#    x_Ib = ((m[i+1]-m[i])+(m[i]-m[i-1]))/2
#    d_Ib = np.append(d_Ib, x_Ib)
#    y_Is = ((n[i+1]-n[i])+(n[i]-n[i-1]))/2
#    d_Is = np.append(d_Is, y_Is)
#    for j in range(len(n)):
#        xy_dPdI = R_sh*m[j]*(-(n[j]/m[j])/(d_Is[j]/d_Ib[j]))
#        dPdI = np.append(dPdI, xy_dPdI)
    

n1A = len(n_1) - 50
n2A = len(n_1) - 40
n3A = len(n_1) - 30
n4A = len(n_1) - 20
n5A = len(n_1) - 10
n6A = len(n_1) - 1

n1B = len(n_2) - 50
n2B = len(n_2) - 40
n3B = len(n_2) - 30
n4B = len(n_2) - 20
n5B = len(n_2) - 10
n6B = len(n_2) - 1

n1C = len(n_3) - 50
n2C = len(n_3) - 40
n3C = len(n_3) - 30
n4C = len(n_3) - 20
n5C = len(n_3) - 10
n6C = len(n_3) - 1

n1D = len(n_4) - 50
n2D = len(n_4) - 40
n3D = len(n_4) - 30
n4D = len(n_4) - 20
n5D = len(n_4) - 10
n6D = len(n_4) - 1

n1E = len(n_5) - 50
n2E = len(n_5) - 40
n3E = len(n_5) - 30
n4E = len(n_5) - 20
n5E = len(n_5) - 10
n6E = len(n_5) - 1

n1A_2 = len(n_6) - 50
n2A_2 = len(n_6) - 40
n3A_2 = len(n_6) - 30
n4A_2 = len(n_6) - 20
n5A_2 = len(n_6) - 10
n6A_2 = len(n_6) - 1


n1B_2 = len(n_7) - 50
n2B_2 = len(n_7) - 40
n3B_2 = len(n_7) - 30
n4B_2 = len(n_7) - 20
n5B_2 = len(n_7) - 10
n6B_2 = len(n_7) - 1

n1C_2 = len(n_8) - 50
n2C_2 = len(n_8) - 40
n3C_2 = len(n_8) - 30
n4C_2 = len(n_8) - 20
n5C_2 = len(n_8) - 10
n6C_2 = len(n_8) - 1

n1D_2 = len(n_9) - 50
n2D_2 = len(n_9) - 40
n3D_2 = len(n_9) - 30
n4D_2 = len(n_9) - 20
n5D_2 = len(n_9) - 10
n6D_2 = len(n_9) - 1

n1E_2 = len(n_10) - 50
n2E_2 = len(n_10) - 40
n3E_2 = len(n_10) - 30
n4E_2 = len(n_10) - 20
n5E_2 = len(n_10) - 10
n6E_2 = len(n_10) - 1

n1F_2 = len(n_11) - 50
n2F_2 = len(n_11) - 40
n3F_2 = len(n_11) - 30
n4F_2 = len(n_11) - 20
n5F_2 = len(n_11) - 10
n6F_2 = len(n_11) - 1

#On the superconducting area the current divides through two branches Ib and Is and we'll consider R tes =0 and we know 

slope1A = (n_1[n2A]-n_1[n1A])/(m_1[m2A]-m_1[m1A])
slope2A = (n_1[n3A]-n_1[n2A])/(m_1[m3A]-m_1[m2A])
slope3A = (n_1[n4A]-n_1[n3A])/(m_1[m4A]-m_1[m3A])
slope4A = (n_1[n5A]-n_1[n4A])/(m_1[m5A]-m_1[m4A])
slope5A = (n_1[n6A]-n_1[n5A])/(m_1[m6A]-m_1[m5A])
slope_initial = np.array([])
list = np.array([slope1A, slope2A, slope3A, slope4A, slope5A])
slope_finalA = np.append(slope_initial, list)

slope1B = (n_2[n2B]-n_2[n1B])/(m_2[m2B]-m_2[m1B])
slope2B = (n_2[n3B]-n_2[n2B])/(m_2[m3B]-m_2[m2B])
slope3B = (n_2[n4B]-n_2[n3B])/(m_2[m4B]-m_2[m3B])
slope4B = (n_2[n5B]-n_2[n4B])/(m_2[m5B]-m_2[m4B])
slope5B = (n_2[n6B]-n_2[n5B])/(m_2[m6B]-m_2[m5B])
slope_initial = np.array([])
list = np.array([slope1B, slope2B, slope3B, slope4B, slope5B])
slope_finalB = np.append(slope_initial, list)

slope1C = (n_3[n2C]-n_3[n1C])/(m_3[m2C]-m_3[m1C])
slope2C = (n_3[n3C]-n_3[n2C])/(m_3[m3C]-m_3[m2C])
slope3C = (n_3[n4C]-n_3[n3C])/(m_3[m4C]-m_3[m3C])
slope4C = (n_3[n5C]-n_3[n4C])/(m_3[m5C]-m_3[m4C])
slope5C = (n_3[n6C]-n_3[n5C])/(m_3[m6C]-m_3[m5C])
slope_initial = np.array([])
list = np.array([slope1C, slope2C, slope3C, slope4C, slope5C])
slope_finalC = np.append(slope_initial, list)


slope1D = (n_4[n2D]-n_4[n1D])/(m_4[m2D]-m_4[m1D])
slope2D = (n_4[n3D]-n_4[n2D])/(m_4[m3D]-m_4[m2D])
slope3D = (n_4[n4D]-n_4[n3D])/(m_4[m4D]-m_4[m3D])
slope4D = (n_4[n5D]-n_4[n4D])/(m_4[m5D]-m_4[m4D])
slope5D = (n_4[n6D]-n_4[n5D])/(m_4[m6D]-m_4[m5D])
slope_initial = np.array([])
list = np.array([slope1D, slope2D, slope3D, slope4D, slope5D])
slope_finalD = np.append(slope_initial, list)

slope1E = (n_5[n2E]-n_5[n1E])/(m_5[m2E]-m_5[m1E])
slope2E = (n_5[n3E]-n_5[n2E])/(m_5[m3E]-m_5[m2E])
slope3E = (n_5[n4E]-n_5[n3E])/(m_5[m4E]-m_5[m3E])
slope4E = (n_5[n5E]-n_5[n4E])/(m_5[m5E]-m_5[m4E])
slope5E = (n_5[n6E]-n_5[n5E])/(m_5[m6E]-m_5[m5E])
slope_initial = np.array([])
list = np.array([slope1E, slope2E, slope3E, slope4E, slope5E])
slope_finalE = np.append(slope_initial, list)

slope1A_2 = (n_6[n2A_2]-n_6[n1A_2])/(m_6[m2A_2]-m_6[m1A_2])
slope2A_2 = (n_6[n3A_2]-n_6[n2A_2])/(m_6[m3A_2]-m_6[m2A_2])
slope3A_2 = (n_6[n4A_2]-n_6[n3A_2])/(m_6[m4A_2]-m_6[m3A_2])
slope4A_2 = (n_6[n5A_2]-n_6[n4A_2])/(m_6[m5A_2]-m_6[m4A_2])
slope5A_2 = (n_6[n6A_2]-n_6[n5A_2])/(m_6[m6A_2]-m_6[m5A_2])
slope_initial = np.array([])
list = np.array([slope1A_2, slope2A_2, slope3A_2, slope4A_2, slope5A_2])
slope_finalA_2 = np.append(slope_initial, list)

slope1B_2 = (n_7[n2B_2]-n_7[n1B_2])/(m_7[m2B_2]-m_7[m1B_2])
slope2B_2 = (n_7[n3B_2]-n_7[n2B_2])/(m_7[m3B_2]-m_7[m2B_2])
slope3B_2 = (n_7[n4B_2]-n_7[n3B_2])/(m_7[m4B_2]-m_7[m3B_2])
slope4B_2 = (n_7[n5B_2]-n_7[n4B_2])/(m_7[m5B_2]-m_7[m4B_2])
slope5B_2 = (n_7[n6B_2]-n_7[n5B_2])/(m_7[m6B_2]-m_7[m5B_2])
slope_initial = np.array([])
list = np.array([slope1B_2, slope2B_2, slope3B_2, slope4B_2, slope5B_2])
slope_finalB_2 = np.append(slope_initial, list)

slope1C_2 = (n_8[n2C_2]-n_8[n1C_2])/(m_8[m2C_2]-m_8[m1C_2])
slope2C_2 = (n_8[n3C_2]-n_8[n2C_2])/(m_8[m3C_2]-m_8[m2C_2])
slope3C_2 = (n_8[n4C_2]-n_7[n3C_2])/(m_8[m4C_2]-m_8[m3C_2])
slope4C_2 = (n_8[n5C_2]-n_8[n4C_2])/(m_8[m5C_2]-m_8[m4C_2])
slope5C_2 = (n_8[n6C_2]-n_8[n5C_2])/(m_8[m6C_2]-m_8[m5C_2])
slope_initial = np.array([])
list = np.array([slope1C_2, slope2C_2, slope3C_2, slope4C_2, slope5C_2])
slope_finalC_2 = np.append(slope_initial, list)

slope1D_2 = (n_9[n2D_2]-n_9[n1D_2])/(m_9[m2D_2]-m_9[m1D_2])
slope2D_2 = (n_9[n3D_2]-n_9[n2D_2])/(m_9[m3D_2]-m_9[m2D_2])
slope3D_2 = (n_9[n4D_2]-n_9[n3D_2])/(m_9[m4D_2]-m_9[m3D_2])
slope4D_2 = (n_9[n5D_2]-n_9[n4D_2])/(m_9[m5D_2]-m_9[m4D_2])
slope5D_2 = (n_9[n6D_2]-n_9[n5D_2])/(m_9[m6D_2]-m_9[m5D_2])
slope_initial = np.array([])
list = np.array([slope1D_2, slope2D_2, slope3D_2, slope4D_2, slope5D_2])
slope_finalD_2 = np.append(slope_initial, list)

slope1E_2 = (n_10[n2E_2]-n_10[n1E_2])/(m_10[m2E_2]-m_10[m1E_2])
slope2E_2 = (n_10[n3E_2]-n_10[n2E_2])/(m_10[m3E_2]-m_10[m2E_2])
slope3E_2 = (n_10[n4E_2]-n_10[n3E_2])/(m_10[m4E_2]-m_10[m3E_2])
slope4E_2 = (n_10[n5E_2]-n_10[n4E_2])/(m_10[m5E_2]-m_10[m4E_2])
slope5E_2 = (n_10[n6E_2]-n_10[n5E_2])/(m_10[m6E_2]-m_10[m5E_2])
slope_initial = np.array([])
list = np.array([slope1E_2, slope2E_2, slope3E_2, slope4E_2, slope5E_2])
slope_finalE_2 = np.append(slope_initial, list)

slope1F_2 = (n_11[n2F_2]-n_11[n1F_2])/(m_11[m2F_2]-m_11[m1F_2])
slope2F_2 = (n_11[n3F_2]-n_11[n2F_2])/(m_11[m3F_2]-m_11[m2F_2])
slope3F_2 = (n_11[n4F_2]-n_11[n3F_2])/(m_11[m4F_2]-m_11[m3F_2])
slope4F_2 = (n_11[n5F_2]-n_11[n4F_2])/(m_11[m5F_2]-m_11[m4F_2])
slope5F_2 = (n_11[n6F_2]-n_11[n5F_2])/(m_11[m6F_2]-m_11[m5F_2])
slope_initial = np.array([])
list = np.array([slope1E_2, slope2E_2, slope3E_2, slope4E_2, slope5E_2])
slope_finalF_2 = np.append(slope_initial, list)


sumA = 0
for i in range(len(slope_finalA)):
    sumA = sumA + slope_finalA[i]
slopeA = sumA/len(slope_finalA)

sumB = 0
for i in range(len(slope_finalB)):
    sumB = sumB + slope_finalB[i]
slopeB = sumB/len(slope_finalB)

sumC = 0
for i in range(len(slope_finalC)):
    sumC = sumC + slope_finalC[i]
slopeC = sumC/len(slope_finalC)

sumD = 0
for i in range(len(slope_finalD)):
    sumD = sumD + slope_finalD[i]
slopeD = sumD/len(slope_finalD)

sumE = 0
for i in range(len(slope_finalE)):
    sumE = sumE + slope_finalE[i]
slopeE = sumE/len(slope_finalE)

sumA_2 = 0
for i in range(len(slope_finalA_2)):
    sumA_2 = sumA_2 + slope_finalA_2[i]
slopeA_2 = sumA_2/len(slope_finalA_2)

sumB_2 = 0
for i in range(len(slope_finalB_2)):
    sumB_2 = sumB_2 + slope_finalB_2[i]
slopeB_2 = sumB_2/len(slope_finalB_2)

sumC_2 = 0
for i in range(len(slope_finalC_2)):
    sumC_2 = sumC_2 + slope_finalC_2[i]
slopeC_2 = sumC_2/len(slope_finalC_2)

sumD_2 = 0
for i in range(len(slope_finalD_2)):
    sumD_2 = sumD_2 + slope_finalD_2[i]
slopeD_2 = sumD_2/len(slope_finalD_2)

sumE_2 = 0
for i in range(len(slope_finalE_2)):
    sumE_2 = sumE_2 + slope_finalE_2[i]
slopeE_2 = sumE_2/len(slope_finalE_2)

sumF_2 = 0
for i in range(len(slope_finalF_2)):
    sumF_2 = sumF_2 + slope_finalF_2[i]
slopeF_2 = sumF_2/len(slope_finalF_2)

for i in range(len(m_1_new)):
    for j in range(len(n_1_new)):
        intercpA = n_1_new[j]-slopeA*m_1_new[i]
        
for i in range(len(m_2_new)):
    for j in range(len(n_2_new)):
        intercpB = n_2_new[j]-slopeB*m_2_new[i]
        
for i in range(len(m_3_new)):
    for j in range(len(n_3_new)):
        intercpC = n_3_new[j]-slopeC*m_3_new[i]
        
for i in range(len(m_4_new)):
    for j in range(len(n_4_new)):
        intercpD = n_4_new[j]-slopeD*m_4_new[i]

for i in range(len(m_5_new)):
    for j in range(len(n_5_new)):
        intercpE = n_5_new[j]-slopeE*m_5_new[i]
        
for i in range(len(m_6_new)):
    for j in range(len(n_6_new)):
        intercpA_2 = n_6_new[j]-slopeA_2*m_6_new[i]

for i in range(len(m_7_new)):
    for j in range(len(n_7_new)):
        intercpB_2 = n_7_new[j]-slopeB_2*m_7_new[i]
        
for i in range(len(m_8_new)):
    for j in range(len(n_8_new)):
        intercpC_2 = n_8_new[j]-slopeC_2*m_8_new[i]
        
for i in range(len(m_9_new)):
    for j in range(len(n_9_new)):
        intercpD_2 = n_9_new[j]-slopeD_2*m_9_new[i]
        
for i in range(len(m_10_new)):
    for j in range(len(n_10_new)):
        intercpE_2 = n_10_new[j]-slopeE_2*m_10_new[i]
        
for i in range(len(m_11_new)):
    for j in range(len(n_11_new)):
        intercpF_2 = n_11_new[j]-slopeF_2*m_11_new[i]
#intercp is the offset


offset = 108.541276
# don't know how to use the intercept function in python and I used the intercept function in excel to calculate the offset


for i in range(len(n_1_new)):
    n_1_new[i] = intercpA - n_1_new[i]

for i in range(len(n_2_new)):
    n_2_new[i] = intercpA - n_2_new[i]

for i in range(len(n_3_new)):
    n_3_new[i] = intercpC - n_3_new[i]
    
for i in range(len(n_4_new)):
    n_4_new[i] = intercpD - n_4_new[i]

for i in range(len(n_5_new)):
    n_5_new[i] = intercpE - n_5_new[i]

for i in range(len(n_6_new)):
    n_6_new[i] = intercpA_2 - n_6_new[i]
    
for i in range(len(n_7_new)):
    n_7_new[i] = intercpB_2 - n_7_new[i]

for i in range(len(n_8_new)):
    n_8_new[i] = intercpC_2 - n_8_new[i]
    
for i in range(len(n_9_new)):
    n_9_new[i] = intercpD_2 - n_9_new[i]
    
for i in range(len(n_10_new)):
    n_10_new[i] = intercpE_2 - n_10_new[i]
    
for i in range(len(n_11_new)):
    n_11_new[i] = intercpF_2 - n_11_new[i]
#calculating the adjusted Is(uA) and plotting it

#plt.plot(m_1_new, n_1_new, 'o', color='black', label = 'Side1pA')
#plt.plot(m_2_new, n_2_new, 'o', color='pink', label = 'Side1pB')
#plt.plot(m_3_new, n_3_new, 'o', color='red', label = 'Side1pC')
#plt.plot(m_4_new, n_4_new, 'o', color='blue', label = 'Side1pD')
#plt.plot(m_5_new, n_5_new, 'o', color='yellow', label = 'Side1pE')
#plt.plot(m_6_new, n_6_new, 'o', color='orange', label = 'Side2pA')
#plt.plot(m_7_new, n_7_new, 'o', color='purple', label = 'Side2pB')
#plt.plot(m_8_new, n_8_new, 'o', color='green', label = 'Side2pC')
#plt.plot(m_9_new, n_9_new, 'o', color='magenta', label = 'Side2pD')
#plt.plot(m_10_new, n_10_new, 'o', color='teal', label = 'Side2pE')
#plt.plot(m_11_new, n_11_new, 'o', color='chocolate', label = 'Side2pF')
#plt.legend(["Side1pA","Side1pB", "Side1pC", "Side1pD", "Side1pE", "Side2pA", "Side2pB", "Side2pC", "Side2pD", "Side2pE", "Side2pF"], loc=0 )
#plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#plt.plot(figsize=(8, 12))
#plt.title("Ib-Is after correction")
#plt.xlabel('Ib(uA)',fontsize=14)
#plt.ylabel('Is_corrected(uA)',fontsize=14)
#Plot Is(adjusted) - Ib


m_1A_max = m_1_new[0]-2
n_1A_max = (n_1_new[0]/n_1_new[1])*1.020*n_1_new[0]
R_sh = 8 
Rp_1A_max = (m_1_new[0]/(2*n_1_new[0])-1)*R_sh

m_1B_max = m_2_new[0]-2
n_1B_max = (n_2_new[0]/n_2_new[1])*1.020*n_2_new[0]
R_sh = 8 
Rp_1B_max = (m_2_new[0]/(2*n_2_new[0])-1)*R_sh

m_1C_max = m_3_new[0]-2
n_1C_max = (n_3_new[0]/n_3_new[1])*1.020*n_3_new[0]
R_sh = 8 
Rp_1C_max = (m_3_new[0]/(2*n_3_new[0])-1)*R_sh

m_1D_max = m_4_new[0]-2
n_1D_max = (n_4_new[0]/n_4_new[1])*1.020*n_4_new[0]
R_sh = 8 
Rp_1D_max = (m_4_new[0]/(2*n_4_new[0])-1)*R_sh

m_1E_max = m_5_new[0]-2
n_1E_max = (n_5_new[0]/n_5_new[1])*1.020*n_5_new[0]
R_sh = 8 
Rp_1E_max = (m_5_new[0]/(2*n_5_new[0])-1)*R_sh

m_2A_max = m_6_new[0]-2
n_2A_max = (n_6_new[0]/n_6_new[1])*1.020*n_6_new[0]
R_sh = 8 
Rp_2A_max = (m_6_new[0]/(2*n_6_new[0])-1)*R_sh

m_2B_max = m_7_new[0]-2
n_2B_max = (n_7_new[0]/n_7_new[1])*1.020*n_7_new[0]
R_sh = 8 
Rp_2B_max = (m_7_new[0]/(2*n_7_new[0])-1)*R_sh

m_2C_max = m_8_new[0]-2
n_2C_max = (n_8_new[0]/n_8_new[1])*1.020*n_8_new[0]
R_sh = 8 
Rp_2C_max = (m_8_new[0]/(2*n_8_new[0])-1)*R_sh

m_2D_max = m_9_new[0]-2
n_2D_max = (n_9_new[0]/n_9_new[1])*1.020*n_9_new[0]
R_sh = 8 
Rp_2D_max = (m_9_new[0]/(2*n_9_new[0])-1)*R_sh

m_2E_max = m_10_new[0]-2
n_2E_max = (n_10_new[0]/n_10_new[1])*1.020*n_10_new[0]
R_sh = 8 
Rp_2E_max = (m_10_new[0]/(2*n_10_new[0])-1)*R_sh

m_2F_max = m_11_new[0]-2
n_2F_max = (n_11_new[0]/n_11_new[1])*1.020*n_11_new[0]
R_sh = 8 
Rp_2F_max = (m_11_new[0]/(2*n_11_new[0])-1)*R_sh

Rp_1A_min_estimated = (m_1A_max/(2*n_1A_max)-1)*R_sh
Rp_1A_estimated = (Rp_1A_min_estimated + Rp_1A_max)/2

Rp_1B_min_estimated = (m_1B_max/(2*n_1B_max)-1)*R_sh
Rp_1B_estimated = (Rp_1B_min_estimated + Rp_1B_max)/2

Rp_1C_min_estimated = (m_1C_max/(2*n_1C_max)-1)*R_sh
Rp_1C_estimated = (Rp_1C_min_estimated + Rp_1C_max)/2

Rp_1D_min_estimated = (m_1D_max/(2*n_1D_max)-1)*R_sh
Rp_1D_estimated = (Rp_1D_min_estimated + Rp_1D_max)/2

Rp_1E_min_estimated = (m_1E_max/(2*n_1E_max)-1)*R_sh
Rp_1E_estimated = (Rp_1E_min_estimated + Rp_1E_max)/2

Rp_2A_min_estimated = (m_2A_max/(2*n_2A_max)-1)*R_sh
Rp_2A_estimated = (Rp_2A_min_estimated + Rp_2A_max)/2

Rp_2B_min_estimated = (m_2B_max/(2*n_2B_max)-1)*R_sh
Rp_2B_estimated = (Rp_2B_min_estimated + Rp_2B_max)/2

Rp_2C_min_estimated = (m_2C_max/(2*n_2C_max)-1)*R_sh
Rp_2C_estimated = (Rp_2C_min_estimated + Rp_2C_max)/2

Rp_2D_min_estimated = (m_2D_max/(2*n_2D_max)-1)*R_sh
Rp_2D_estimated = (Rp_2D_min_estimated + Rp_2D_max)/2

Rp_2E_min_estimated = (m_2E_max/(2*n_2E_max)-1)*R_sh
Rp_2E_estimated = (Rp_2E_min_estimated + Rp_2E_max)/2

Rp_2F_min_estimated = (m_2F_max/(2*n_2F_max)-1)*R_sh
Rp_2F_estimated = (Rp_2F_min_estimated + Rp_2F_max)/2

Rtes_1A_estimated = np.array([])
for i in range(len(n_1_new)):
    x = ((m_1_new[i]-n_1_new[i])/n_1_new[i])*(R_sh)-Rp_1A_estimated  
    Rtes_1A_estimated = np.append(Rtes_1A_estimated, x)

Rtes_1B_estimated = np.array([])
for i in range(len(n_2_new)):
    x = ((m_2_new[i]-n_2_new[i])/n_2_new[i])*(R_sh)-Rp_1B_estimated  
    Rtes_1B_estimated = np.append(Rtes_1B_estimated, x)
    
Rtes_1C_estimated = np.array([])
for i in range(len(n_3_new)):
    x = ((m_3_new[i]-n_3_new[i])/n_3_new[i])*(R_sh)-Rp_1C_estimated  
    Rtes_1C_estimated = np.append(Rtes_1C_estimated, x)
    
Rtes_1D_estimated = np.array([])
for i in range(len(n_4_new)):
    x = ((m_4_new[i]-n_4_new[i])/n_4_new[i])*(R_sh)-Rp_1D_estimated  
    Rtes_1D_estimated = np.append(Rtes_1D_estimated, x)
    
Rtes_1E_estimated = np.array([])
for i in range(len(n_5_new)):
    x = ((m_5_new[i]-n_5_new[i])/n_5_new[i])*(R_sh)-Rp_1E_estimated  
    Rtes_1E_estimated = np.append(Rtes_1E_estimated, x)
    
Rtes_2A_estimated = np.array([])
for i in range(len(n_6_new)):
    x = ((m_6_new[i]-n_6_new[i])/n_6_new[i])*(R_sh)-Rp_2A_estimated  
    Rtes_2A_estimated = np.append(Rtes_2A_estimated, x)
    
Rtes_2B_estimated = np.array([])
for i in range(len(n_7_new)):
    x = ((m_7_new[i]-n_7_new[i])/n_7_new[i])*(R_sh)-Rp_2B_estimated  
    Rtes_2B_estimated = np.append(Rtes_2B_estimated, x)

Rtes_2C_estimated = np.array([])
for i in range(len(n_8_new)):
    x = ((m_8_new[i]-n_8_new[i])/n_8_new[i])*(R_sh)-Rp_2C_estimated  
    Rtes_2C_estimated = np.append(Rtes_2C_estimated, x)
    
Rtes_2D_estimated = np.array([])
for i in range(len(n_9_new)):
    x = ((m_9_new[i]-n_9_new[i])/n_9_new[i])*(R_sh)-Rp_2D_estimated  
    Rtes_2D_estimated = np.append(Rtes_2D_estimated, x)
    
Rtes_2E_estimated = np.array([])
for i in range(len(n_10_new)):
    x = ((m_10_new[i]-n_10_new[i])/n_10_new[i])*(R_sh)-Rp_2E_estimated  
    Rtes_2E_estimated = np.append(Rtes_2E_estimated, x)
    
Rtes_2F_estimated = np.array([])
for i in range(len(n_11_new)):
    x = ((m_11_new[i]-n_11_new[i])/n_11_new[i])*(R_sh)-Rp_2F_estimated  
    Rtes_2F_estimated = np.append(Rtes_2F_estimated, x)

Ptes_1A_estimated = np.array([])
for i in range(len(n_1_new)):
    y = (n_1_new[i]*n_1_new[i]*Rtes_1A_estimated[i])/1000
    Ptes_1A_estimated = np.append(Ptes_1A_estimated,y)

Ptes_1B_estimated = np.array([])
for i in range(len(n_2_new)):
    y = (n_2_new[i]*n_2_new[i]*Rtes_1B_estimated[i])/1000
    Ptes_1B_estimated = np.append(Ptes_1B_estimated,y)
    
Ptes_1C_estimated = np.array([])
for i in range(len(n_3_new)):
    y = (n_3_new[i]*n_3_new[i]*Rtes_1C_estimated[i])/1000
    Ptes_1C_estimated = np.append(Ptes_1C_estimated,y)
    
Ptes_1D_estimated = np.array([])
for i in range(len(n_4_new)):
    y = (n_4_new[i]*n_4_new[i]*Rtes_1D_estimated[i])/1000
    Ptes_1D_estimated = np.append(Ptes_1D_estimated,y)
    
Ptes_1E_estimated = np.array([])
for i in range(len(n_5_new)):
    y = (n_5_new[i]*n_5_new[i]*Rtes_1E_estimated[i])/1000
    Ptes_1E_estimated = np.append(Ptes_1E_estimated,y)
    
Ptes_2A_estimated = np.array([])
for i in range(len(n_6_new)):
    y = (n_6_new[i]*n_6_new[i]*Rtes_2A_estimated[i])/1000
    Ptes_2A_estimated = np.append(Ptes_2A_estimated,y)
    
Ptes_2B_estimated = np.array([])
for i in range(len(n_7_new)):
    y = (n_7_new[i]*n_7_new[i]*Rtes_2B_estimated[i])/1000
    Ptes_2B_estimated = np.append(Ptes_2B_estimated,y)
    
Ptes_2C_estimated = np.array([])
for i in range(len(n_8_new)):
    y = (n_8_new[i]*n_8_new[i]*Rtes_2C_estimated[i])/1000
    Ptes_2C_estimated = np.append(Ptes_2C_estimated,y)
    
Ptes_2D_estimated = np.array([])
for i in range(len(n_9_new)):
    y = (n_9_new[i]*n_9_new[i]*Rtes_2D_estimated[i])/1000
    Ptes_2D_estimated = np.append(Ptes_2D_estimated,y)
    
Ptes_2E_estimated = np.array([])
for i in range(len(n_10_new)):
    y = (n_10_new[i]*n_10_new[i]*Rtes_2E_estimated[i])/1000
    Ptes_2E_estimated = np.append(Ptes_2E_estimated,y)
    
Ptes_2F_estimated = np.array([])
for i in range(len(n_11_new)):
    y = (n_11_new[i]*n_11_new[i]*Rtes_2F_estimated[i])/1000
    Ptes_2F_estimated = np.append(Ptes_2F_estimated,y)
    
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
plt.savefig('Ib-Is after offset.jpg')
plt.savefig('output.png', dpi=300, bbox_inches='tight')
    
#plt.plot(m_1_new, Ptes_1A_estimated,  color='black', label = 'Side1pA', linewidth = 2.7)
#plt.plot(m_2_new, Ptes_1B_estimated, color='pink', label = 'Side1pB', linewidth = 2.7)
#plt.plot(m_3_new, Ptes_1C_estimated, color='red', label = 'Side1pC', linewidth = 2.7)
#plt.plot(m_4_new, Ptes_1D_estimated, color='blue', label = 'Side1pD', linewidth = 2.7)
#plt.plot(m_5_new, Ptes_1E_estimated, color='yellow', label = 'Side1pE', linewidth = 2.7)
#plt.plot(m_6_new, Ptes_2A_estimated, color='orange', label = 'Side2pA', linewidth = 2.7)
#plt.plot(m_7_new, Ptes_2B_estimated, color='purple', label = 'Side2pB', linewidth = 2.7)
#plt.plot(m_8_new, Ptes_2C_estimated, color='green', label = 'Side2pC', linewidth = 2.7)
#plt.plot(m_9_new, Ptes_2D_estimated, color='magenta', label = 'Side2pD', linewidth = 2.7)
#plt.plot(m_10_new, Ptes_2E_estimated, color='teal', label = 'Side2pE', linewidth = 2.7)
#plt.plot(m_11_new, Ptes_2F_estimated, color='chocolate', label = 'Side2pF', linewidth = 2.7)
#plt.title("TES Power versus Ib")
#plt.xlabel('Ib(uA)',fontsize=14)
#plt.ylabel('P_tes',fontsize=14)
#plt.legend(["Side1pA","Side1pB", "Side1pC", "Side1pD", "Side1pE", "Side2pA", "Side2pB", "Side2pC", "Side2pD", "Side2pE", "Side2pF"], loc=0 )
#plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

#TES power versus Ib plot

plt.plot(m_1_new, Rtes_1A_estimated,  color='black', label = 'Side1pA', linewidth = 2.7)
plt.plot(m_2_new, Rtes_1B_estimated, color='pink', label = 'Side1pB', linewidth = 2.7)
plt.plot(m_3_new, Rtes_1C_estimated, color='red', label = 'Side1pC', linewidth = 2.7)
plt.plot(m_4_new, Rtes_1D_estimated, color='blue', label = 'Side1pD', linewidth = 2.7)
plt.plot(m_5_new, Rtes_1E_estimated, color='yellow', label = 'Side1pE', linewidth = 2.7)
plt.plot(m_6_new, Rtes_2A_estimated, color='orange', label = 'Side2pA', linewidth = 2.7)
plt.plot(m_7_new, Rtes_2B_estimated, color='purple', label = 'Side2pB', linewidth = 2.7)
plt.plot(m_8_new, Rtes_2C_estimated, color='green', label = 'Side2pC', linewidth = 2.7)
plt.plot(m_9_new, Rtes_2D_estimated, color='magenta', label = 'Side2pD', linewidth = 2.7)
plt.plot(m_10_new, Rtes_2E_estimated, color='teal', label = 'Side2pE', linewidth = 2.7)
plt.plot(m_11_new, Rtes_2F_estimated, color='chocolate', label = 'Side2pF', linewidth = 2.7)
plt.title("TES Resistance versus Ib")
plt.xlabel('Ib(uA)',fontsize=14)
plt.ylabel('R_tes',fontsize=14)
plt.legend(["Side1pA","Side1pB", "Side1pC", "Side1pD", "Side1pE", "Side2pA", "Side2pB", "Side2pC", "Side2pD", "Side2pE", "Side2pF"], loc=0 )
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

#TES resistance versus Ib plot
