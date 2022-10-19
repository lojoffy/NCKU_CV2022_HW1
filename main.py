import time
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import glob, os

from torchsummary import summary
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import cv2
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from model import VGG16
from model import VGG16_bn

import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.models import resnet50

from torch.utils.tensorboard import SummaryWriter

from PIL import Image
import imageio

dataset_prefix = "/Users/blueshiner/git/3_Collections/29_CV_HW1/"
folder_1 = dataset_prefix + "dataset/Q1_Image/"
folder_2 = dataset_prefix + "dataset/Q2_Image/"
folder_3 = dataset_prefix + "dataset/Q3_Image/"

# ------- init setting -------
st.set_page_config(
    page_title="NCKU CV&DL HW1 - F14071075",
    page_icon="random",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.title('NCKU - Computer Vision and Deep Learing 2022')
st.subheader("電腦視覺與深度學習 Homework 1")
# ------- end -------

# ----- Warning Part -----
st.caption("Github: [Github link](https://github.com/RoyChao19477/NCKU_CV2022_HW1)")
st.caption("Author: F14071075@2022")
# ----- end -----

# tab1, tab2 = st.tabs(["Cat", "Dog"])
# with tab1:
#     st.write("Sec 1")
# with tab2:
#     st.write("Sec 2")

# select box:
topic = st.selectbox("Select an assignment",
        (
            '(1) Camera Calibration',
            '(2) Augmented Reality',
            '(3) Stereo Disparity Map',
            '(5) Training Cifar-10 Classifier Using VGG16')
        )

# Stable Variances:
if 'state_1' not in st.session_state:
    st.session_state['state_1'] = 0
if 'state_3' not in st.session_state:
    st.session_state['state_3'] = 0
if 'state_5' not in st.session_state:
    st.session_state['state_5'] = 0


# used function
def no_fn():
    pass

# ------- HW1 - 1 -------
if topic == '(1) Camera Calibration':
    st.header("Camera Calibration")
    
    # ------- Upload a Picture -------
    col_1_00, col_1_01 = st.columns(2)
    with col_1_00:
        image_0 = st.file_uploader("Upload Image", type=['jpg', 'png', 'bmp'])
        if image_0 is not None:
            file_bytes = np.asarray(bytearray(image_0.read()), dtype=np.uint8)
            cv_image_0 = cv2.imdecode(file_bytes, 1)

    with col_1_01:
        idx_0_0 = st.slider('or use default dataset', 1, 15)
        f_name = folder_1 + str(idx_0_0) + ".bmp"
        cv_image_0 = cv2.imread(f_name)

    if st.session_state.state_1 >= 5:
        age2 = st.slider('Select an image num', 1, 15)

    if cv_image_0 is not None: 
        col_1_0, col_1_1 = st.columns(2)
        with col_1_0:
            st.write(cv_image_0.shape)
            st.image(cv_image_0, channels="BGR")

        with col_1_1:
            if st.button("Corner Detection :"):
                st.session_state.state_1 = 1
            if st.session_state.state_1 >= 1:
                ret, corners = cv2.findChessboardCorners(cv_image_0, (11,8), None)      # ret: detection result (True/False), corners: collection of corners (multiple pair of dots)
                if ret:
                    for pt in corners:
                        point = pt[0]
                        # print(point)
                        cv2.circle(cv_image_0, center=(int(point[0]), int(point[1])), radius=10, color=(0, 0, 255), thickness=-1)   # draw dots (circle)
                    #st.write("ret", ret)
                    #st.write("corner", corners)
                    cv2.drawChessboardCorners(cv_image_0,(11,8),corners,ret)    # draw lines on image
                    st.image(cv_image_0, channels="BGR")

        #col_1_2, col_1_3, col_1_4 = st.columns(3)
        #with col_1_2:
        if st.button("Intrinsic Matrix :"):
            st.session_state.state_1 = 2
        if st.session_state.state_1 >= 2:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objpoints = []  # 3d points in real world space
            imgpoints = []  # 2d points in image plane
            
            objp = np.zeros((1, 11 * 8, 3), np.float32)     # Note: our chessboard is 12 * 9, (0:11, 0:8)
            objp[0,:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
            prev_img_shape = None
            
            #images = glob.glob("/home/roy/roy/10_CV2/image15/*.bmp")
            images = glob.glob( folder_1 + "*.bmp")
            for im in images:
                img = cv2.imread(im)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # ret: detection result (True/False), corners: collection of corners (
                ret, corners = cv2.findChessboardCorners(gray, (11, 8), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ret:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
                    imgpoints.append(corners2)
                    img2 = cv2.drawChessboardCorners(cv_image_0, (11, 8), corners2, ret)

            h,w = img2.shape[:2]
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            st.write(mtx)

        #with col_1_3:
        if st.button("Extrinsic Matrix :"):
            st.session_state.state_1 = 3
        if st.session_state.state_1 >= 3:
            age = st.slider('Select an image', 1, 15)
            _rvec = rvecs[age - 1]
            _tvec = tvecs[age - 1]
            _rmtx = cv2.Rodrigues(rvecs[age - 1])
            e_mtx = np.concatenate((_rmtx[0], _tvec), axis=1)
            st.write(e_mtx)

        #with col_1_4:
            if st.button("Distortion Matrix :"):
                st.session_state.state_1 = 4
            if st.session_state.state_1 >= 4:
                st.write(dist)

        if st.button("Show Undistorted Result :"):
            st.session_state.state_1 = 5
        if st.session_state.state_1 >= 5:
            age2 = st.slider('Select an image index', 1, 15)
            f_name = folder_1 + str(age2) + ".bmp"
            imgg = cv2.imread(f_name)
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            img3 = cv2.undistort(imgg, mtx, dist, None, newcameramtx)
            col_1_5, col_1_6 = st.columns(2)
            with col_1_5:
                st.write("Original")
                st.image(imgg, channels="BGR")
            with col_1_6:
                st.write("Later")
                st.image(img3, channels="BGR")

# ------- end -------


# ------- HW1 - 2 -------
if topic == '(2) Augmented Reality':
    st.header("Augmented Reality")

    word = st.text_input('Input an word (no more than 6 characters)', 'INPUT')
    if len(word) > 6:
        st.error("The WORD should not be more than 6 characters.")
    elif len(word) <= 0:
        st.error("The WORD shold not be less than 0 characters.")
    else:
        word = word.upper()
        st.write('The current word is ', word)
        fs = cv2.FileStorage(folder_2 + 'Q2_lib/' + 'alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        fs_v = cv2.FileStorage(folder_2 + 'Q2_lib/' + 'alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        #ch = fs.getNode('K').mat()


        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        
        objp = np.zeros((1, 11 * 8, 3), np.float32)     # Note: our chessboard is 12 * 9, (0:11, 0:8)
        objp[0,:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        prev_img_shape = None
        # ret: detection result (True/False), corners: collection of corners

        images = glob.glob( folder_2 + "*.bmp")
        for im in images:
            img = cv2.imread(im)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
                imgpoints.append(corners2)
                img2 = cv2.drawChessboardCorners(img, (11, 8), corners2, ret)

        h,w = img2.shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


        age2_1 = st.slider('Select an image number', 0, len(images) - 1)
        st.write("Image: ", str( images[age2_1] ))

        to_be_draw = []
        img2_1 = cv2.imread( images[age2_1] )
        _rvec = rvecs[age2_1]
        _tvec = tvecs[age2_1].reshape(3,1)

        position_mapping = [
            [7,5,0],
            [4,5,0],
            [1,5,0],
            [7,2,0],
            [4,2,0],
            [1,2,0]  
        ]

        # FLAT version
        img4 = img2_1.copy()
        for iter_word_idx in range(len(word)):
            ch = fs.getNode( word[iter_word_idx] ).mat()
            for iter_ch in ch:
                ach = np.float32([position_mapping[iter_word_idx], position_mapping[iter_word_idx]])
                iter_ch = np.add(iter_ch, ach)
                imagePoints, _ = cv2.projectPoints(iter_ch, _rvec, _tvec, mtx, dist)
                to_be_draw.append( imagePoints )
        
        for iter_line in to_be_draw:
            img4 = cv2.line(img4, tuple(iter_line.reshape(2, 2)[0].astype(int)), tuple(iter_line.reshape(2,2)[1].astype(int)), (0, 0, 255), 20, cv2.LINE_AA)

        col_2_1, col_2_2 = st.columns(2)
        with col_2_1:
            st.image(img2_1.copy(), channels="BGR")
        with col_2_2:
            st.image(img4, channels="BGR")

        to_be_draw = []
        # VERTICLE version
        img5 = img2_1.copy()
        for iter_word_idx in range(len(word)):
            ch = fs_v.getNode( word[iter_word_idx] ).mat()
            for iter_ch in ch:
                ach = np.float32([position_mapping[iter_word_idx], position_mapping[iter_word_idx]])
                iter_ch = np.add(iter_ch, ach)
                imagePoints, _ = cv2.projectPoints(iter_ch, _rvec, _tvec, mtx, dist)
                to_be_draw.append( imagePoints )
        
        for iter_line in to_be_draw:
            img5 = cv2.line(img5, tuple(iter_line.reshape(2, 2)[0].astype(int)), tuple(iter_line.reshape(2,2)[1].astype(int)), (0, 0, 255), 20, cv2.LINE_AA)

        col_2_3, col_2_4 = st.columns(2)
        with col_2_3:
            st.image(img2_1.copy(), channels="BGR")
        with col_2_4:
            st.image(img5, channels="BGR")


# ------- end -------


# ------- HW1 - 3 -------
if topic == '(3) Stereo Disparity Map':
    st.header("Stereo Disparity Map")
    cv_image_3_1, cv_image_3_2 = None, None

    col_3_1, col_3_2, col_3_3 = st.columns(3)
    with col_3_1:
        image_3_1 = st.file_uploader("Upload imL image", type=['jpg', 'png', 'bmp'])
        if image_3_1 is not None:
            file_bytes = np.asarray(bytearray(image_3_1.read()), dtype=np.uint8)
            cv_image_3_1 = cv2.imdecode(file_bytes, 1)

    with col_3_2:
        image_3_2 = st.file_uploader("Upload imR image", type=['jpg', 'png', 'bmp'])
        if image_3_2 is not None:
            file_bytes = np.asarray(bytearray(image_3_2.read()), dtype=np.uint8)
            cv_image_3_2 = cv2.imdecode(file_bytes, 1)

    with col_3_3:
        st.write("Or you can load default images here. (imL and imR)")
        if st.button('Load default dataset'):
            st.session_state.state_3 = 1
        if st.session_state.state_3 >= 1:
            cv_image_3_1 = cv2.imread(folder_3 + "imL.png")
            cv_image_3_2 = cv2.imread(folder_3 + "imR.png")
            st.success("Default imL.png and imR.png were loaded successfully !")
    
    col_3_4, col_3_5 = st.columns(2)
    with col_3_4:
        st.image(cv_image_3_1, channels="BGR")
    with col_3_5:
        st.image(cv_image_3_2, channels="BGR")

    if st.button("Calculate Disparity (Normalized to 0 ~ 255)"):
        st.session_state.state_3 = 2
    if st.session_state.state_3 >= 2:
        imgL_gray = cv2.cvtColor(cv_image_3_1, cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.cvtColor(cv_image_3_2, cv2.COLOR_BGR2GRAY)
        disparity = cv2.StereoBM_create(256, 25).compute(imgL_gray, imgR_gray)

        # Normalize to 0 ~ 255
        shift_disp = disparity - np.min(disparity)
        norm_disparity = (shift_disp / np.max(shift_disp) * 255).astype(int)
        st.image(norm_disparity)

    if st.button("Checking the Disparity Value"):
        st.session_state.state_3 = 3
    if st.session_state.state_3 >= 3:
        h,w = cv_image_3_1.shape[:2]
        col_3_6, col_3_7 = st.columns(2)
        with col_3_6: 
            idx_x = st.slider('Select X-axis position', 0, w)
        with col_3_7: 
            idx_y = st.slider('Select Y-axis position', 0, h)
        while (norm_disparity[idx_y, idx_x] == 0):
            idx_x += 1
        st.write("Now X-axis is :", idx_x)

        col_3_8, col_3_9 = st.columns(2)
        with col_3_8: 
            img_ref = cv2.circle(cv_image_3_1.copy(), (idx_x, idx_y), radius=20, color=(0, 0, 255), thickness=-1)
            img_ref = cv2.circle(img_ref, (idx_x, idx_y), radius=50, color=(0, 0, 255), thickness=5)
            st.image(img_ref, channels="BGR")
        with col_3_9: 
            img_map = cv2.circle(cv_image_3_2.copy(), (idx_x - norm_disparity[idx_y, idx_x], idx_y), radius=20, color=(0, 0, 255), thickness=-1)
            img_map = cv2.circle(img_map, (idx_x - norm_disparity[idx_y, idx_x], idx_y), radius=50, color=(0, 0, 255), thickness=5)
            st.image(img_map, channels="BGR")
        st.write(f"Disparity value at ({idx_x}, {idx_y}) is : {norm_disparity[idx_y, idx_x]}")

# ------- end -------


# ------- HW1 - 5 -------
if topic == '(5) Training Cifar-10 Classifier Using VGG16':
    st.header("Training Cifar-10 Classifier Using VGG16")

    # ------- Upload a Picture -------

    if st.button("Load PyTorch:"):
        st.session_state.state_5 = 1
    if st.session_state.state_5 >= 1:
        st.write("Loading...")
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        batch_size = 1
        trainset = torchvision.datasets.CIFAR10(root='VGG16/CIFAR-10', train=True,
                                                        download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                          shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='VGG16/CIFAR-10', train=False,
                                                       download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                         shuffle=False, num_workers=2)
        classes = ('plane', 'car', 'bird', 'cat',
                           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # get some random training images
        st.write("Finished!")
        
        if st.button("Show Samples:"):
            dataiter = iter(testloader)
            for i in range(9):
                images, labels = dataiter.next()
                image = images / 2 + 0.5
                image = image / image.max() * 255
                image = image.squeeze(0).permute(1, 2, 0).repeat_interleave(4, dim=0).repeat_interleave(4, dim=1).numpy().astype(int)
                st.write("This is: ", classes[labels.item()])
                st.image(image)
        
        if st.button("VGG16 hyperparameters:"):
            st.session_state.state_5 = 2
        if st.session_state.state_5 >= 2:
            st.text("VGG16 hparameters: ")
            st.text("  batch size:     32")
            st.text("  learning rate;  0.0001")
            st.text("  optimizer:      Adam")
        
            st.text("VGG16 BatchNorm2dhparameters: ")
            st.text("  batch size:     32")
            st.text("  learning rate;  0.0001")
            st.text("  optimizer:      Adam")
     
        if st.button("Model Structure: VGG16"):
            st.code( summary(VGG16(), (3, 64, 64)) )
        
        if st.button("Model Structure: VGG16 with BatchNorm2d"):
            st.code( summary(VGG16_bn(), (3, 64, 64)) )

        if st.button("Loss and Accurancy:"):
            st.session_state.state_5 = 3
        if st.session_state.state_5 >= 3:
            st.write("VGG16 100epoch")
            train_loss = np.loadtxt("VGG16/train_loss.csv", delimiter=',')
            test_loss = np.loadtxt("VGG16/test_loss.csv", delimiter=',')
            st.line_chart( 
                    pd.DataFrame( 
                        np.transpose( np.stack((train_loss, test_loss)), (1, 0)),
                        columns=['train loss', 'test loss']) 
                    )
            
            st.write("VGG16 BatchNorm2d 100epoch")
            train_loss2 = np.loadtxt("VGG16/train_loss_bn.csv", delimiter=',')
            test_loss2 = np.loadtxt("VGG16/test_loss_bn.csv", delimiter=',')
            st.line_chart( 
                    pd.DataFrame( 
                        np.transpose( np.stack((train_loss2, test_loss2)), (1, 0)),
                        columns=['train loss', 'test loss']) 
                    )
            train_loss3 = np.loadtxt("VGG16/train_accu_bn.csv", delimiter=',')
            test_loss3 = np.loadtxt("VGG16/test_accu_bn.csv", delimiter=',')
            st.line_chart( 
                    pd.DataFrame( 
                        np.transpose( np.stack((train_loss3, test_loss3)), (1, 0)),
                        columns=['train Accu', 'test Accu']) 
                    )
        
        if st.button("Use model to predict:"):
            st.session_state.state_5 = 4
        if st.session_state.state_5 >= 4:
            path = "VGG16/VGG16_epoch100.pt"
            model = VGG16()
            device = torch.device('cpu')
            model.to(device)
            cpt = torch.load(path, map_location=device)
            model.load_state_dict(cpt)
            model.eval()
            
            path = "VGG16/VGG16bn_epoch100.pt"
            model2 = VGG16_bn()
            device = torch.device('cpu')
            model2.to(device)
            cpt = torch.load(path, map_location=device)
            model2.load_state_dict(cpt)
            model2.eval()


            dataiter = iter(trainloader)
            images, labels = dataiter.next()
            image = images / 2 + 0.5
            image = image / image.max() * 255
            image = image.squeeze(0).permute(1, 2, 0).repeat_interleave(4, dim=0).repeat_interleave(4, dim=1).numpy().astype(int)
            st.write("Ground Truth: ", classes[labels.item()])
            st.image(image)
         
            st.write("Pretrained VGG16 BatchNorm2d:")
            pred = model(images)
            st.code(pred)
            _, predicted = torch.max(pred.data, 1)
            st.code( classes[predicted.item()] )

            st.write("Pretrained VGG16 BatchNorm2d:")
            pred2 = model2(images)
            st.code(pred2)
            _, predicted2 = torch.max(pred2.data, 1)
            st.code( classes[predicted2.item()] )

    # ------- end -------

# ------- end -------
# ------- end -------

# ------ Torch ------
