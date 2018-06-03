import numpy as np
import cv2

templates = [
    {
        'width' : 1128,  
        'height': 984+100,
        'kpts' : {"nose":[530, 208], 
                "l_eye": [570, 166], "r_eye":[490, 166],
                "l_ear": [612, 206], "r_ear": [440, 208],
                "l_shoulder": [612, 338], "r_shoulder": [440, 338],
                "l_elbow": [762, 338], "r_elbow": [278, 338],
                "l_wrist": [884, 338], "r_wrist": [130, 338],
                "l_hip": [612, 610], "r_hip": [440, 610],
                "l_knee": [686, 726-20], "r_knee": [366, 726-20],
                "l_ankle": [762, 850-40], "r_ankle": [280, 850-40],
                "head": [530, 86], "neck": [530, 378], "neck-c": [530, 338]
                 }
    },
    {
        'width' : 368,  
        'height': 368,
        'kpts' : {"nose": [187, 94],
                  "l_eye": [198, 84], "r_eye": [175, 85],
                  "l_ear": [209, 117], "r_ear": [163, 119],
                  "l_shoulder": [251, 155], "r_shoulder": [124, 156],
                  "l_elbow": [263, 248], "r_elbow": [108, 248], 
                  "l_wrist": [225, 247], "r_wrist": [144, 244],
                  "l_hip": [221, 309], "r_hip": [153, 310], 
                  "neck-c": [187.5, 155.5]
                 }
    }  
]
def templates17to29(templates17):
    inter = {
        'neck-c':['l_shoulder', 'r_shoulder'], 
        'hip-c':['l_hip', 'r_hip'], 
        'l_body-c':['l_shoulder', 'l_hip'], 
        'r_body-c':['r_shoulder', 'r_hip'], 
        'l_upperarm-c':['l_shoulder', 'l_elbow'], 
        'r_upperarm-c':['r_shoulder', 'r_elbow'], 
        'l_lowerarm-c':['l_elbow', 'l_wrist'], 
        'r_lowerarm-c':['r_elbow', 'r_wrist'], 
        'l_upperleg-c':['l_hip', 'l_knee'], 
        'r_upperleg-c':['r_hip', 'r_knee'], 
        'l_lowerleg-c':['l_knee', 'l_ankle'], 
        'r_lowerleg-c':['r_knee', 'r_ankle'], 
    }
    templates29 = templates17
    for i, templ in enumerate(templates29):
        for key_inter, key_pairs in inter.items():
            if key_pairs[0] in templ['kpts'].keys() and key_pairs[1] in templ['kpts'].keys():
                templates29[i]['kpts'][key_inter] = [(x1+x2)/2.0 for x1, x2 in zip(templ['kpts'][key_pairs[0]], templ['kpts'][key_pairs[1]])]
    return templates29

templates = templates17to29(templates)

class Template:
    def __init__(self, template_width, template_height, template_kpts):
        self.template_width = float(template_width)
        self.template_height = float(template_height)
        self.template_kpts = template_kpts.copy()
        self.part_names = template_kpts.keys()
        self.np = len(self.part_names)
    
    def resize(self, width, height):
        ratio_w = width / self.template_width
        ratio_h = height / self.template_height
        for key, cor in self.template_kpts.items():
            self.template_kpts[key] = [cor[0]*ratio_w, cor[1]*ratio_h]

class PoseAffineTemplate:
    def __init__(self, npart, width, height):
        self.npart = npart # 18 or 29
        self.template_width = width
        self.template_height = height
        self.template_list = self._load(templates, width, height)
        
        
    def _load(self, ts, w, h):
        template_list = []
        for t in ts:
            template = Template(t['width'], t['height'], t['kpts'])
            template.resize(w, h)
            template_list.append(template)
        return template_list
    
    def drawKpts(self, img, kpts, npart, circle = 10):
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                      [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255]]
        try:
            keypoints = np.array(kpts).reshape(npart,3)
        except:
            keypoints = np.array(kpts).reshape(npart,2)
        canvas = img.copy()
        for i, p in enumerate(keypoints):
            if p[0]==0 and p[1]==0:
                continue
            cv2.circle(canvas, (int(p[0]), int(p[1])), circle, colors[i%len(colors)], thickness=-1)
        to_plot = cv2.addWeighted(img, 0.3, canvas, 0.7, 0)
        return to_plot

    def drawTemplates(self, circle=10):
        to_plots = []
        for template in self.template_list:
            kpts = []
            for key in template.part_names:
                kpts.append(template.template_kpts[key])
            canvas = np.zeros((self.template_height, self.template_width, 3), dtype = np.uint8)
            to_plot = self.drawKpts(canvas, kpts, template.np, circle=10)
            to_plots.append(to_plot)
        return to_plots

    def _estimateH(self, template, src_kpts, imgwidth, imheight, flip=False):
        if self.npart == 18:
            src_part_names = ["nose", 'neck-c', 
                       "r_shoulder", "r_elbow", "r_wrist",
                       "l_shoulder", "l_elbow", "l_wrist",
                       "r_hip", "r_knee", "r_ankle",
                       "l_hip", "l_knee", "l_ankle",
                       "r_eye", "l_eye", "r_ear", "l_ear"]
        elif self.npart == 29:
            src_part_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear',
                             'l_shoulder', 'neck-c', 'r_shoulder',
                             'l_upperarm-c', 'r_upperarm-c',
                             'l_elbow', 'r_elbow',
                             'l_lowerarm-c', 'r_lowerarm-c',
                             'l_wrist', 'r_wrist',
                             'l_body-c', 'r_body-c',
                             'l_hip', 'hip-c', 'r_hip',
                             'l_upperleg-c', 'r_upperleg-c',
                             'l_knee', 'r_knee',
                             'l_lowerleg-c', 'r_lowerleg-c'
                             'l_ankle', 'r_ankle']
            
        keypoints = np.array(src_kpts).reshape(-1,3)
        src_pts = []
        dst_pts = []
        outliners = 0
        for name, p in zip(src_part_names, keypoints):
            if name not in template.part_names:
                if p[0]==0 and p[1]==0:
                    continue
                else:
                    outliners+=10
                    continue
            if p[0]==0 and p[1]==0:
                outliners+=1
                continue
            templatename = name
            if flip:
                if name[0:2]=='r_':
                    templatename = templatename.replace('r_', 'l_')
                elif name[0:2]=='l_':
                    templatename = templatename.replace('l_', 'r_')
            dst_pts.append(template.template_kpts[templatename])
            src_pts.append([p[0], p[1]])
        
        if len(src_pts)==0:
            return np.array([[1,  0., 0], 
                             [-0., 1, 0]]), 999999999
        
        src_pts = np.float32(src_pts).reshape(-1,1,2)
        dst_pts = np.float32(dst_pts).reshape(-1,1,2)
        
        # special case: only on keypoints. just trans and resize. no rotate. eg(4820)
        if len(src_pts)==1:
            #print 'special case: len of src_pts is 1.'
            scale = np.mean([float(self.template_width)/imgwidth, 
                             float(self.template_height)/imheight])/2
            delta = scale*src_pts - dst_pts
            H = np.array([[scale,  0., -delta[0,0,0]], 
                          [-0., scale, -delta[0,0,1]]])
            return H, 0 + outliners*500
        # generate case
        try:
            H = cv2.estimateRigidTransform(src_pts, dst_pts, fullAffine=False)
            if H is None:
                #print 'use local affine'
                H = self._localAffineEstimate(src_pts, dst_pts, fullAfine=False)
        except:
            print (src_pts, dst_pts, src_kpts)
        
        # calc error
        src_cor = src_pts.reshape(-1,2).transpose() # (2, N)
        src_cor = np.concatenate((src_cor, np.ones((1, src_cor.shape[1]), dtype=np.float32)), axis=0) # (3, N)
        error = H.dot(src_cor)-dst_pts.reshape(-1,2).transpose()  # (2, N)
        error1 = np.mean(np.abs(error)) + outliners*500
        #error2 = np.linalg.norm(error,ord=1)
        return H, error1
    
    def estimateH(self, src_kpts, imgwidth, imheight):
        bestH = None
        bestError = 999999999999
        bestT = None
        for template in self.template_list:
            H, err = self._estimateH(template, src_kpts, imgwidth, imheight, flip=False)
            if err<bestError:
                bestError = err
                bestH = H
                bestT = template
            H_flip, err_flip = self._estimateH(template, src_kpts, imgwidth, imheight, flip=True)
            if err_flip<bestError:
                bestError = err_flip
                bestH = H_flip
                bestT = template
        # print bestT.np, bestError
        
        minx, miny, maxx, maxy = cal(bestH, imheight, imgwidth, self.template_height, self.template_width)
        sx = self.template_width/(maxx-minx)
        sy = self.template_height/(maxy-miny)
        dx = - minx*sx
        dy = - miny*sy
        H_aug = np.array([[sx,  0., dx], 
                          [-0., sy, dy]])
        bestH = H_aug.dot(np.concatenate((bestH, np.array([0,0,1]).reshape((1,3))), axis = 0))
        return bestH
    
    def _localAffineEstimate(self, src, dst, fullAfine=False):
        '''
        Document: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=solve#cv2.solve
        C++ Version: aff_trans.cpp in opencv
        src: numpy array (N, 1, 2)
        dst: numpy array (N, 1, 2)
        '''
        out = np.zeros((2,3), np.float32)
        siz = 2*src.shape[0]

        if fullAfine:
            matM = np.zeros((siz,6), np.float32)
            matP = np.zeros((siz,1), np.float32)
            contPt=0
            for ii in range(0, siz):
                therow = np.zeros((1,6), np.float32)
                if ii%2==0:
                    therow[0,0] = src[contPt, 0, 0] # x
                    therow[0,1] = src[contPt, 0, 1] # y
                    therow[0,2] = 1
                    matM[ii,:] = therow[0,:].copy()
                    matP[ii,0] = dst[contPt, 0, 0] # x
                else:
                    therow[0,3] = src[contPt, 0, 0] # x
                    therow[0,4] = src[contPt, 0, 1] # y
                    therow[0,5] = 1
                    matM[ii,:] = therow[0,:].copy()
                    matP[ii,0] = dst[contPt, 0, 1] # y
                    contPt += 1
            
            sol = cv2.solve(matM, matP, flags = cv2.DECOMP_SVD)
            sol = sol[1]
            out = sol.reshape(2, -1)
        else:
            matM = np.zeros((siz,4), np.float32)
            matP = np.zeros((siz,1), np.float32)
            contPt=0
            for ii in range(0, siz):
                therow = np.zeros((1,4), np.float32)
                if ii%2==0:
                    therow[0,0] = src[contPt, 0, 0] # x
                    therow[0,1] = src[contPt, 0, 1] # y
                    therow[0,2] = 1
                    matM[ii,:] = therow[0,:].copy()
                    matP[ii,0] = dst[contPt, 0, 0] # x
                else:
                    therow[0,0] = src[contPt, 0, 1] # y ## Notice, c++ version is - here
                    therow[0,1] = -src[contPt, 0, 0] # x
                    therow[0,3] = 1
                    matM[ii,:] = therow[0,:].copy()
                    matP[ii,0] = dst[contPt, 0, 1] # y
                    contPt += 1
            sol = cv2.solve(matM, matP, flags = cv2.DECOMP_SVD)
            sol = sol[1]
            out[0,0]=sol[0,0]
            out[0,1]=sol[1,0]
            out[0,2]=sol[2,0]
            out[1,0]=-sol[1,0]
            out[1,1]=sol[0,0]
            out[1,2]=sol[3,0]

        # result
        return out

    
def jun(x, y, xt, yt):
    countpo = 0
    countne = 0
    for i in range(4):
        res = np.cross([x[(i+1)%4]-x[i], y[(i+1)%4]-y[i]], [xt-x[i], yt-y[i]])
        if res > 0:countpo += 1
        elif res < 0:countne += 1
        else:continue
    if countpo == 0 or countne == 0:return True
    else:return False
        

def point(x1,y1, x2,y2, x1n,y1n, x2n,y2n):
    eps = 1e-5
    a1 = y1 - y2
    b1 = x2 - x1
    c1 = y1 * (x1 - x2) -x1 * (y1 -y2)
    a2 = y1n - y2n
    b2 = x2n - x1n
    c2 = y1n * (x1n - x2n) -x1n * (y1n -y2n)
    if (a1 * b2 - a2 * b1) == 0:
        if (c2 * b1 - c1 * b2) == 0:
            x1_ = min(x1, x2)
            x2_ = max(x1, x2)
            x1n_ = min(x1n, x2n)
            x2n_ = max(x1n, x2n)
            max_x = max(x1_, x1n_)
            min_x = min(x2_, x2n_)
            y1_ = min(y1, y2)
            y2_ = max(y1, y2)
            y1n_ = min(y1n, y2n)
            y2n_ = max(y1n, y2n)
            max_y = max(y1_, y1n_)
            min_y = min(y1_, y1n_)
            if max_x > min_x:
                return []
            else:return [[max_x, max_y], [min_x, min_y]]
        else:return []
    else:
        x = (c2 * b1 - c1 * b2)/(a1 * b2 - a2 * b1)
        y = (c1 * a2 - c2 * a1)/(a1 * b2 - a2 * b1)
        if x <= max(x1, x2)+eps and x >= min(x1, x2)-eps and y <= max(y1, y2)+eps and y >= min(y1, y2)-eps:
            if x <= max(x1n, x2n)+eps and x >= min(x1n, x2n)-eps and y <= max(y1n, y2n)+eps and y >= min(y1n, y2n)-eps:
                return [[x, y]]
            else:return []
        else:return []

def cal(W, height, width, height_t, width_t):
    '''
    W: (2, 3) same with H.
    height, width: before Affine by W, the size of image.
    height_t, width_t: Affine dst size.
    '''
    x1 = np.array([0, 0, 1])
    x2 = np.array([width, 0, 1])
    x3 = np.array([width, height, 1])
    x4 = np.array([0, height, 1])
    x1t = W.dot(x1)
    x2t = W.dot(x2)
    x3t = W.dot(x3)
    x4t = W.dot(x4)
    x = [0, width_t, width_t, 0]
    y = [0, 0, height_t, height_t]
    xt = [x1t[0], x2t[0], x3t[0], x4t[0]]
    yt = [x1t[1], x2t[1], x3t[1], x4t[1]]
    allpoint = []
    for i in range(4):
        for j in range(4):
            result = point(x[j], y[j], x[(j+1)%4], y[(j+1)%4], xt[i], yt[i], xt[(i+1)%4], yt[(i+1)%4])
            for res in result:
                allpoint.append(res)
        if jun(x, y, xt[i], yt[i]):allpoint.append([xt[i], yt[i]])
        if jun(xt, yt, x[i], y[i]):allpoint.append([x[i], y[i]])
    if len(allpoint)==0:
        return 0,0,width_t,height_t
    allpoint = np.array(allpoint)
    maxx, maxy = np.max(allpoint,axis=0)
    minx, miny = np.min(allpoint,axis=0)
    return minx, miny, maxx, maxy
