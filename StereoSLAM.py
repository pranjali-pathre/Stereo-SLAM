import numpy as np 
import cv2
import open3d as o3d
import os
from matplotlib import pyplot as plt
import random
from PIL import Image

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

## Reading poses file
def read_poses(filename='../data/poses.txt'):
    f = open(filename, 'r')
    lines = f.readlines()
    T_21 = []
    for i in range(len(lines)):
        t = lines[i].split()
        T = [] 
        T.append( [float(t[0]), float(t[1]),float(t[2]),float(t[3])] ) 
        T.append( [float(t[4]), float(t[5]),float(t[6]),float(t[7])] ) 
        T.append( [float(t[8]), float(t[9]),float(t[10]),float(t[11])]) 
        T.append([0,0,0,1])
        T_21.append(np.array(T))
    T_21 = np.asarray(T_21)
    return T_21     

## Reading calibration file
def read_calib(filename='../data/calib.txt'):
    f = open(filename, 'r')
    lines = f.readlines()
    k = lines[1].split()
    K = [] 
    K.append( [float(k[0]), float(k[1]),float(k[2])] ) 
    K.append( [float(k[3]), float(k[4]),float(k[5])] ) 
    K.append( [float(k[6]), float(k[7]),float(k[8])] )
    K = np.asarray(K)
    b = lines[4].split()
    B = float(b[0]) 
    return K, B


# In[3]:


left_dir = '../data/img2/' 
right_dir = '../data/img3/'
img_names = sorted(os.listdir(left_dir))
num_images = len(img_names)
TransMat_21 = read_poses()
K, b = read_calib()
focal_length = K[0][0]
final_world_pts = []
final_colors = []
depth_maps_3D = []
depth_maps_2D = []
Q_values = []


# In[4]:


for n in range(num_images):
    left_img = cv2.imread(left_dir + img_names[n])
    right_img = cv2.imread(right_dir + img_names[n])
    window_size = 5
    min_disp = -39
    num_disp = 144
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    disp12MaxDiff = 1,
    blockSize = 5,
    P1 = 8 * 3 * window_size ** 2,
    P2 = 32 * 3 * window_size ** 2,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    preFilterCap=63
    )    
    disparity_map = stereo.compute(left_img,right_img).astype(np.float32) / 64.0    
    disparity_map = (disparity_map-min_disp)/num_disp
    depth_maps_2D.append(disparity_map)
    
    n1, n2 = disparity_map.shape
    
    colors = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    # Setting Q matrix 
    Q = np.asarray([[1, 0, 0, -0.5*n2],[0,-1, 0,  0.5*n1], [0, 0, 0, focal_length], [0, 0, 1/b,  0]])
    
    image_pts = []
    for i in range(n1):
        for j in range(n2):
            image_pts.append(np.array([j,i,disparity_map[i][j],1]))
    image_pts = np.array(image_pts)

    points_3D = []
    for pt in image_pts:
        point = Q@pt
        points_3D.append(point/point[3])
    points_3D = np.asarray(points_3D) 
    depth_maps_3D.append(points_3D)
    
    points_world = TransMat_21[n]@points_3D.T
    
    mask = disparity_map >= disparity_map.min()
    colors = colors[mask]
    colors = colors / 255
    for i in range(len(points_world[0])):
        if(points_world[3][i]>0):
            final_world_pts.append([points_world[0][i]/points_world[3][i],points_world[1][i]/points_world[3][i],points_world[2][i]/points_world[3][i]])
            final_colors.append(colors[i])
final_world_pts = np.asarray(final_world_pts)       
final_colors = np.asarray(final_colors) 


# In[5]:


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(final_world_pts)
pcd.colors = o3d.utility.Vector3dVector(final_colors)
o3d.visualization.draw_geometries([pcd])

final_colors_ = final_colors*255
file_name = 'test.ply'
write_ply(file_name, final_world_pts, final_colors_)

### getting the ground truth
Q1 = np.array([ [-9.1e-01, 5.5e-02, -4.2e-01, -1.9e+02],
               [4.2e-02, 9.983072e-01, 4.2e-02, 1.7e+00],
               [4.2e-01, 2.1e-02, -9.2e-01, 5.5e+01],
              [0,0,0,1]])
Q1 = np.linalg.inv(Q1)
Q = Q1[:3,:]
P_gt = K@Q
print(P_gt)


# Getting correspondances
ind = range(0,10000)
points_3d = []
points_2d = []
ones = np.ones((len(final_world_pts),1))
final_world_pts_homo = np.concatenate((final_world_pts,ones),axis = 1 ) 
proj_gt = P_gt@(final_world_pts_homo.T)
proj_gt = proj_gt/proj_gt[2,:]
proj_gt = proj_gt.T

for i in ind:
    points_3d.append(final_world_pts_homo[i])
    points_2d.append(proj_gt[i])
points_3d = np.array(points_3d)
points_2d = np.array(points_2d)
print(points_2d)


# In[8]:


### Initialising P: 1.Random 2.DLT
def DLT(x,X):
    M = [];
    zeros = np.zeros(4);
    L = len(X);
    for i in range(L):
        M.append(np.hstack((-X[i],np.hstack((zeros,x[i][0]*X[i])))));
        M.append(np.hstack((zeros,np.hstack((-X[i],x[i][1]*X[i])))));
    M = np.array(M).reshape(2*L,12);
    U,D,VT= np.linalg.svd(M);
    P = np.array(VT[-1]).reshape(3,4);
    P = P/P[-1,-1]
    return P/P[-1,-1]
print("The Matrix P is found through DLT is: ")
P_dlt = DLT(points_2d[0:2000],points_3d[0:2000])*P_gt[2,3]
print(P_dlt)


def Guass_Newton(P_est, points_3d, points_2d, itr, tol):
    curr_itr = 0
    flg = 1
    lr = 1
    while (flg==1):
        jac = []
        points_est = []
        P_final = P_est
        for pt in points_3d:
            pp = P_final @ pt
            j = [pt[0]/pp[2],pt[1]/pp[2],pt[2]/pp[2],pt[3]/pp[2],0,0,0,0,(-pp[0]*pt[0])/(pp[2]*pp[2]),-pp[0]*pt[1]/(pp[2]*pp[2]),-pp[0]*pt[2]/(pp[2]*pp[2]),-pp[0]*pt[3]/(pp[2]*pp[2])]
            jac.append(np.array(j))
            j = [0,0,0,0,pt[0]/(pp[2]),pt[1]/(pp[2]),pt[2]/(pp[2]),pt[3]/(pp[2]),-pp[1]*pt[0]/(pp[2]*pp[2]),-pp[1]*pt[1]/(pp[2]*pp[2]),-pp[1]*pt[2]/(pp[2]*pp[2]),-pp[1]*pt[3]/(pp[2]*pp[2])]
            jac.append(np.array(j))
            pp = pp/pp[2]
            points_est.append(pp.T)
        jac = np.array(jac)
        points_est = np.array(points_est)
        er = []
        for i in range(len(points_est)):
            er.append(points_2d[i][0]-points_est[i][0])
            er.append(points_2d[i][1]-points_est[i][1])
        er = np.array(er)
        er = er.reshape(2*len(points_est),1)
        gg = np.linalg.pinv(jac.T@jac)
        P_final = P_final.reshape(12,1)
        update = gg@(jac.T@er)
        P_est = P_final + lr*update
        up = P_est - P_final
        diff = np.sqrt(up.T@up)
        P_est = P_est.reshape(3,4)
        P_final = P_final.reshape(3,4)
        er_2 = (er.T@er)
        curr_itr = curr_itr + 1
#         if(curr_itr % 10 == 0):
#             print(curr_itr,"iterations done")
        if(np.linalg.norm(jac.T@er) < tol):
#             print(curr_itr,"iterations done")
            itr_req = curr_itr
            print("Error in estimate: "+ str(diff))
            flg = 0
        if(curr_itr > itr):
#             print(curr_itr,"iterations done")
            print("Error in estimate: "+ str(diff))
            itr_req = curr_itr
            flg = 0
    
    return P_final, er_2, itr_req


# In[10]:


#Initialising DLT output (best initialization)
P_GN, error, itr_req = Guass_Newton(P_dlt, points_3d, points_2d, itr=100, tol=1e-7)
print("Iterations required; ", itr_req)
print("P from Gauss-Newton: \n", P_GN)
print("True P:\n", P_gt)
# print("Error: ",error)


# In[11]:


# Initialisation converging at another local minima
P_est = np.array([[ 3.05963083e+01, -1.86410995e+00,  0.48368729e+00,  5.34986476e+03],
 [ 1.24910530e+00, -2.45765334e+01,  5.16404815e+00, -4.91253523e+00],
 [ 1.41944683e-02, -1.43197526e-03,  3.08978789e-02,  1.00000000e+00]])
P_GN, error, itr_req = Guass_Newton(P_est, points_3d, points_2d, itr=100, tol=1e-7)
print("Iterations required; ", itr_req)
print("P from Gauss-Newton: \n", P_GN)
print("True P:\n", P_gt)

## Far initialisation which won't converge to ground truth 
P_est = np.array([[-7.97699140e-04, -1.56731843e-03, -9.64602229e-04,  5.53706762e-02],
                 [-1.68118921e-04,  4.15993556e-03,  1.04888333e-03,  5.32315076e-02],
                 [-1.05955850e-04, -9.42056054e-04, -2.61847498e-04,  1.00000000e+00]])
P_GN, error, itr_req = Guass_Newton(P_est, points_3d, points_2d, itr=100, tol=1e-7)
print("Iterations required; ", itr_req)
print("P from Gauss-Newton: \n", P_GN)
print("True P:\n", P_gt)



list1 = []
list2 = []

for n in range(num_images - 1):
    query_img = cv2.imread(left_dir + img_names[n])
    train_img = cv2.imread(left_dir + img_names[n+1])  

    query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)

    matcher = cv2.BFMatcher()
    matches = matcher.match(queryDescriptors,trainDescriptors)

    list_kp1 = [queryKeypoints[mat.queryIdx].pt for mat in matches] 
    list_kp2 = [trainKeypoints[mat.trainIdx].pt for mat in matches]
    list1.append(list_kp1)
    list2.append(list_kp2)


def find_cost(p, q):
    r = np.sum(np.linalg.norm(p - q, axis=1)**2)/p.shape[0]
    return r

def ICP(p, q):
    p_ = p - np.mean(p, axis = 0)
    q_ = q - np.mean(q, axis = 0)
    W = np.dot(p_.T, q_)/p.shape[0]
    U, S, VT = np.linalg.svd(W)
    sigma = np.eye(3)
    sigma[2,2] = np.linalg.det(U)*np.linalg.det(VT.T)
    R = np.dot(VT.T,U.T)
    
    if(np.linalg.det(U) < 0):
        R = np.dot(VT.T, np.dot(sigma,U.T)) 
    t = np.mean(q, axis = 0) - R@np.mean(p, axis = 0)
    return R, t

def procrustes_alignment(p, q):
    num_iter = 1
    tol = 1e-15
    q0 = q
    cost = np.zeros((1, num_iter))
    for i in range(num_iter):
        cost[0, i] = find_cost(p, q)
        R, t = ICP(p, q)
        t = np.reshape(t, (3,1))
        p = np.dot(R, p.T) + t
        p = p.T
        if(np.linalg.norm(p - q0) < tol):
            print(np.linalg.norm(p - q0))
            break
    return R, t  

R_ICP = []
t_ICP = []
T_ICP = []
R_curr = np.eye(3)
T_curr = np.eye(4)
for n in range(num_images - 1):
    pc1 = []
    pc2 = []
    for nn in range(500):
        x_l1 = int(list1[n][nn][0])
        x_l2 = int(list2[n][nn][0])
        y_l1 = int(list1[n][nn][1])
        y_l2 = int(list2[n][nn][1])
        pc1.append(depth_maps_3D[n][(y_l1*n2 + x_l1), :3])
        pc2.append(depth_maps_3D[n+1][(y_l2*n2 + x_l2), :3])
    pc1 = np.array(pc1)
    pc2 = np.array(pc2)  
    R, t = procrustes_alignment(pc1, pc2)
    T = np.eye(4)
    T[:3, :3] = R
    t = t.reshape((3,))
    T[:3, 3] = t
    T_curr = T_curr@T
    T_ICP.append(T_curr)    

P_ini = []
T1 = np.eye(4)
P1 = K@T1[:3, :]
P_ini.append(P1)
for i in range(num_images - 1):
    Pini = K@T_ICP[i][:3, :]
    P_ini.append(Pini)
P_est = []

depth_maps_2D_PnP = []
for i in range(num_images):
    o = P_ini[i]@(depth_maps_3D[i][:10000, :].T)
    o = o/o[2,:]
    o = o.T
    depth_maps_2D_PnP.append(o)

for i in range (num_images):
    print("For pose ", i+1, ":")
    P_GN, error, itr_req = Guass_Newton(P_ini[i], depth_maps_3D[i][:5000, :], depth_maps_2D_PnP[i], itr=200, tol=1e-7)
    P_est.append(P_GN)
    print("P from Gauss-Newton: \n", P_GN, "\n")
