import os
import math
import numpy as np
import scipy.io as sio

class morphabelAngle(object):
    def __init__(self):
        filename = "data.mat"
        prefixPath = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(prefixPath,filename)
        self.model = sio.loadmat(model_path)
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.n_exp_para = self.model['expPC'].shape[1]


    def from_image(self,vertices,h,w):
        vertices[:,0] = vertices[:,0] - w/2
        vertices[:,1] = h/2 -1 - vertices[:,1]
        return vertices

    def matrix2angle(self,R):
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi
        return rx, ry, rz

    def estimate_affine_matrix_3d22d(self,X,x):
        X = X.T; x = x.T
        assert(x.shape[1] == X.shape[1])
        n = x.shape[1]
        assert(n >= 4)

        #--- 1. normalization
        # 2d points
        mean = np.mean(x, 1) # (2,)
        x = x - np.tile(mean[:, np.newaxis], [1, n])
        average_norm = np.mean(np.sqrt(np.sum(x**2, 0)))
        scale = np.sqrt(2) / average_norm
        x = scale * x

        T = np.zeros((3,3), dtype = np.float32)
        T[0, 0] = T[1, 1] = scale
        T[:2, 2] = -mean*scale
        T[2, 2] = 1

        # 3d points
        X_homo = np.vstack((X, np.ones((1, n))))
        mean = np.mean(X, 1) # (3,)
        X = X - np.tile(mean[:, np.newaxis], [1, n])
        m = X_homo[:3,:] - X
        average_norm = np.mean(np.sqrt(np.sum(X**2, 0)))
        scale = np.sqrt(3) / average_norm
        X = scale * X

        U = np.zeros((4,4), dtype = np.float32)
        U[0, 0] = U[1, 1] = U[2, 2] = scale
        U[:3, 3] = -mean*scale
        U[3, 3] = 1

        # --- 2. equations
        A = np.zeros((n*2, 8), dtype = np.float32)
        X_homo = np.vstack((X, np.ones((1, n)))).T
        A[:n, :4] = X_homo
        A[n:, 4:] = X_homo
        b = np.reshape(x, [-1, 1])
    
        # --- 3. solution
        p_8 = np.linalg.pinv(A).dot(b)
        P = np.zeros((3, 4), dtype = np.float32)
        P[0, :] = p_8[:4, 0]
        P[1, :] = p_8[4:, 0]
        P[-1, -1] = 1

        # --- 4. denormalization
        P_Affine = np.linalg.inv(T).dot(P.dot(U))
        return P_Affine

    def P2sRt(self,P):
        t = P[:, 3]
        R1 = P[0:1, :3]
        R2 = P[1:2, :3]
        s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
        r1 = R1/np.linalg.norm(R1)
        r2 = R2/np.linalg.norm(R2)
        r3 = np.cross(r1, r2)

        R = np.concatenate((r1, r2, r3), 0)
        return s, R, t

    def estimate_expression(self,x, shapeMU, expPC, expEV, shape, s, R, t2d, lamb = 2000):
        x = x.copy()
        assert(shapeMU.shape[0] == expPC.shape[0])
        assert(shapeMU.shape[0] == x.shape[1]*3)

        dof = expPC.shape[1]

        n = x.shape[1]
        sigma = expEV
        t2d = np.array(t2d)
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
        A = s*P.dot(R)

        # --- calc pc
        pc_3d = np.resize(expPC.T, [dof, n, 3]) 
        pc_3d = np.reshape(pc_3d, [dof*n, 3]) 
        pc_2d = pc_3d.dot(A.T) 
        pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 29

        # --- calc b
        # shapeMU
        mu_3d = np.resize(shapeMU, [n, 3]).T # 3 x n
        # expression
        shape_3d = shape
        # 
        b = A.dot(mu_3d + shape_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
        b = np.reshape(b.T, [-1, 1]) # 2n x 1

        # --- solve
        equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1/sigma**2)
        x = np.reshape(x.T, [-1, 1])
        equation_right = np.dot(pc.T, x - b)

        exp_para = np.dot(np.linalg.inv(equation_left), equation_right)
        
        return exp_para
    
    def estimate_shape(self,x, shapeMU, shapePC, shapeEV, expression, s, R, t2d, lamb = 3000):
        x = x.copy()
        assert(shapeMU.shape[0] == shapePC.shape[0])
        assert(shapeMU.shape[0] == x.shape[1]*3)

        dof = shapePC.shape[1]

        n = x.shape[1]
        sigma = shapeEV
        t2d = np.array(t2d)
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
        A = s*P.dot(R)

        # --- calc pc
        pc_3d = np.resize(shapePC.T, [dof, n, 3]) # 199 x n x 3
        pc_3d = np.reshape(pc_3d, [dof*n, 3]) 
        pc_2d = pc_3d.dot(A.T.copy()) # 199 x n x 2
        
        pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 199

        # --- calc b
        # shapeMU
        mu_3d = np.resize(shapeMU, [n, 3]).T # 3 x n
        # expression
        exp_3d = expression
        # 
        b = A.dot(mu_3d + exp_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
        b = np.reshape(b.T, [-1, 1]) # 2n x 1

        # --- solve
        equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1/sigma**2)
        x = np.reshape(x.T, [-1, 1])
        equation_right = np.dot(pc.T, x - b)

        shape_para = np.dot(np.linalg.inv(equation_left), equation_right)

        return shape_para

    def fit(self,x,max_iter=4):

        x = x.copy().T
        sp = np.zeros((self.n_shape_para, 1), dtype = np.float32)
        ep = np.zeros((self.n_exp_para, 1), dtype = np.float32)
        shapeMU = self.model['shapeMU']
        shapePC = self.model['shapePC']
        expPC = self.model['expPC']

        for i in range(max_iter):
            X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
            X = np.reshape(X, [int(len(X)/3), 3]).T
            P = self.estimate_affine_matrix_3d22d(X.T,x.T)
            s,R,t = self.P2sRt(P)
            shape = shapePC.dot(sp)
            shape = np.reshape(shape, [int(len(shape)/3), 3]).T
            ep = self.estimate_expression(x, shapeMU, expPC, self.model['expEV'][:self.n_exp_para,:], shape, s, R, t[:2], lamb = 20)
            expression = expPC.dot(ep)
            expression = np.reshape(expression, [int(len(expression)/3), 3]).T
            sp = self.estimate_shape(x, shapeMU, shapePC, self.model['shapeEV'][:self.n_shape_para,:], expression, s, R, t[:2], lamb = 40)
        angle = self.matrix2angle(R)
        return angle


    