import os
import tensorflow as tf
import tensorflow.keras as K
import cv2 as cv
import numpy as np
import random

# ====================================================================================================================
# Semi-supervised network based on R-S-Net using ConsInstancy Training
# Details on the R-S-Net: Coenen et al., 2021: Semi-Supervised Segmentation of Concrete Aggregate using Consensus Regularisation and Prior Guidance. In ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences  
def ConsInstancy_Net(input_depth=3, output_depthSeg=1, regularisation=None, doBatchnorm=False):
    inputs = K.Input(shape=(None, None, input_depth))  # static batch-size to enable target loss computation

    # ENCODER
    x = K.layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularisation)(inputs)
    if doBatchnorm:
        x = K.layers.BatchNormalization()(x)
    x1d = K.layers.Activation("relu")(x)

    x2d = R_S_Module_down(x1d, nFilter=32, doBatchnorm=doBatchnorm, regularisation=regularisation)
    x3d = R_S_Module_down(x2d, nFilter=64, doBatchnorm=doBatchnorm, regularisation=regularisation)
    x4d = R_S_Module_down(x3d, nFilter=128, doBatchnorm=doBatchnorm, regularisation=regularisation)
    x5d = R_S_Module_down(x4d, nFilter=256, doBatchnorm=doBatchnorm, regularisation=regularisation)

    
    # _________________________________________________________________________________________________________________
    # Segmentation DECODER Branch
    x4u = R_S_Module_up(x5d, nFilter=256, doBatchnorm=doBatchnorm, regularisation=regularisation)
    x4u = K.layers.concatenate([x4u, x4d])

    x3u = R_S_Module_up(x4u, nFilter=128, doBatchnorm=doBatchnorm, regularisation=regularisation)
    x3u = K.layers.concatenate([x3u, x3d])

    x2u = R_S_Module_up(x3u, nFilter=64, doBatchnorm=doBatchnorm, regularisation=regularisation)
    x2u = K.layers.concatenate([x2u, x2d])

    x1u = R_S_Module_up(x2u, nFilter=32, doBatchnorm=doBatchnorm, regularisation=regularisation)
    x1u = K.layers.concatenate([x1u, x1d])

    x = K.layers.Conv2D(32, 3, activation="relu", padding="same", kernel_regularizer=regularisation)(x1u)
    outSeg = K.layers.Conv2D(output_depthSeg, 1, activation="sigmoid", padding="same", kernel_regularizer=regularisation, name='Segmentation')(x)

    # _________________________________________________________________________________________________________________
    # INSTANCE DECODER Branch

    x4uInst = R_S_Module_up(x5d, nFilter=256, doBatchnorm=doBatchnorm, regularisation=regularisation)
    x4uInst = K.layers.concatenate([x4uInst, x4d])

    x3uInst = R_S_Module_up(x4uInst, nFilter=128, doBatchnorm=doBatchnorm, regularisation=regularisation)
    x3uInst = K.layers.concatenate([x3uInst, x3d])

    x2uInst = R_S_Module_up(x3uInst, nFilter=64, doBatchnorm=doBatchnorm, regularisation=regularisation)
    x2uInst = K.layers.concatenate([x2uInst, x2d])

    x1uInst = R_S_Module_up(x2uInst, nFilter=32, doBatchnorm=doBatchnorm, regularisation=regularisation)
    x1uInst = K.layers.concatenate([x1uInst, x1d])

    x0uOri = K.layers.Conv2D(32, 3, activation="relu", padding="same", kernel_regularizer=regularisation)(x1uInst)
    x0uOri = K.layers.Conv2D(3, 1, activation=None, padding="same", kernel_regularizer=regularisation)(x0uOri)
    outOri = K.layers.Lambda(l2_normalize_layer, name="DirectionNorm")(x0uOri)

    stream = K.layers.concatenate([x1uInst, outOri])

    x0uDist = K.layers.Conv2D(32, 3, activation="relu", padding="same", kernel_regularizer=regularisation)(stream)
    outDist = K.layers.Conv2D(1, 1, activation="sigmoid", padding="same", kernel_regularizer=regularisation, name='DistTrafo')(x0uDist)

    x0uDistInv = K.layers.Conv2D(32, 3, activation="relu", padding="same", kernel_regularizer=regularisation)(stream)
    outDistInv = K.layers.Conv2D(1, 1, activation="sigmoid", padding="same", kernel_regularizer=regularisation, name='DistTrafoInv')(x0uDistInv)

    stream = K.layers.concatenate([x1uInst, outDist, outDistInv])
    x0uBound = K.layers.Conv2D(32, 3, activation="relu", padding="same", kernel_regularizer=regularisation)(stream)
    outBound = K.layers.Conv2D(1, 1, activation="sigmoid", padding="same", kernel_regularizer=regularisation, name='Boundary')(x0uBound)

    model = K.Model(inputs=inputs, outputs=[outOri, outDist, outDistInv, outBound, outSeg])

    return model


# ====================================================================================================================
def R_S_Module_down(x, nFilter, doBatchnorm = False, regularisation=None):
    residual = K.layers.Conv2D(filters=nFilter, kernel_size=3, strides=2, padding="same", kernel_regularizer=regularisation)(x)
    residual =K.layers.Activation("relu")(residual)
    
    x = K.layers.Conv2D(filters=nFilter, kernel_size=3, padding="same", kernel_regularizer=regularisation)(x)
    if doBatchnorm:
        x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)
    
    x = K.layers.SeparableConv2D(filters=nFilter, kernel_size=3, padding="same", kernel_regularizer=regularisation)(x)
    if doBatchnorm:
        x = K.layers.BatchNormalization()(x)   
    x = K.layers.Activation("relu")(x)
    
    x = K.layers.SeparableConv2D(filters=nFilter, kernel_size=3, padding="same", kernel_regularizer=regularisation)(x)
    if doBatchnorm:
        x = K.layers.BatchNormalization()(x)   
    x = K.layers.Activation("relu")(x)
        
    x = K.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)    
    
    x = K.layers.add([x, residual])
        
    return x

# ====================================================================================================================
def R_S_Module_up(x, nFilter, doBatchnorm = False, regularisation=None):
    
    residual = K.layers.Conv2DTranspose(nFilter, (2,2), strides=(2, 2), padding="same", kernel_regularizer=regularisation)(x)
    residual =K.layers.Activation("relu")(residual)
    
    x = K.layers.Conv2D(filters=nFilter, kernel_size=3, padding="same", kernel_regularizer=regularisation)(x)
    
    if doBatchnorm:
        x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)
    
    x = K.layers.SeparableConv2D(filters=nFilter, kernel_size=3, padding="same", kernel_regularizer=regularisation)(x)
    if doBatchnorm:
        x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)
    
    x = K.layers.SeparableConv2D(filters=nFilter, kernel_size=3, padding="same", kernel_regularizer=regularisation)(x)
    if doBatchnorm:
        x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)
    
    x = K.layers.UpSampling2D(2)(x)
    
    x = K.layers.add([x, residual])        
    return x

# ====================================================================================================================
def l2_normalize_layer(x):
    x_norm = tf.math.l2_normalize(x, axis=-1)
    return x_norm

# ====================================================================================================================================================
def generator_ConsInstancy(inputPathImg, inputPathLabel, inputPathImgUnlabeled, in_out_size=(448,448), batchsize=1):    

    # get list of all image names inside the path
    imgList = os.listdir(inputPathImg)
    imgListUnlabeled = os.listdir(inputPathImgUnlabeled)
    
    images = np.zeros(shape=(batchsize*2, in_out_size[1], in_out_size[0], 3), dtype=np.float)
    labelsSeg = np.zeros(shape=(batchsize*2, in_out_size[1], in_out_size[0], 1), dtype=np.float)
    labelsOriMap = np.zeros(shape=(batchsize*2, in_out_size[1], in_out_size[0], 3), dtype=np.float)
    labelsDistTrafo = np.zeros(shape=(batchsize*2, in_out_size[1], in_out_size[0], 1), dtype=np.float)
    labelsDistTrafoInv = np.zeros(shape=(batchsize*2, in_out_size[1], in_out_size[0], 1), dtype=np.float)
    labelsBound = np.zeros(shape=(batchsize*2, in_out_size[1], in_out_size[0], 1), dtype=np.float)
    # infinit loop
    b = 0
    while True:
        
        random.shuffle(imgList)
        
        for i in range(len(imgList)):
            # Supervised BRANCH ____________________________________________
            # Read input image
            fnCurrImg = os.path.join(inputPathImg, imgList[i])
            filename, file_extension = os.path.splitext(imgList[i])
            if not fnCurrImg.endswith(".jpg") and not fnCurrImg.endswith(".png"):
                continue
           
            x = cv.imread(fnCurrImg, cv.IMREAD_COLOR)                           
            x = get_image_sized(x, img_depth=3, out_size=in_out_size)            
            
            # read Seg output
            fnCurrLabelArea = os.path.join(inputPathLabel, 'area', imgList[i])
            yArea = cv.imread(fnCurrLabelArea, cv.IMREAD_GRAYSCALE).astype('float32')
            yArea /= 255.  
            
            # Direction 
            fnCurrDirection = os.path.join(inputPathLabel, 'oriMap', '%s.exr' %(filename))
            yDirection = cv.imread(fnCurrDirection,  cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
            
            # Trafo 
            fnCurrTrafo = os.path.join(inputPathLabel, 'distTrafo', '%s.exr' %(filename))
            yTrafo = cv.imread(fnCurrTrafo,  cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
            
            # TrafoInv 
            fnCurrTrafoInv = os.path.join(inputPathLabel, 'distTrafoInv', '%s.exr' %(filename))
            yTrafoInv = cv.imread(fnCurrTrafoInv,  cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)

            # read Boundary output
            fnCurrLabelBound = os.path.join(inputPathLabel, 'boundary_1px', imgList[i])
            yBound = cv.imread(fnCurrLabelBound, cv.IMREAD_GRAYSCALE).astype('float32')
            yBound /= 255.

            
            # SemiSupervised BRANCH _____________________________________________
            #  Branch input
            idxUnlabeled = np.random.randint(0, len(imgListUnlabeled))
            fnCurrImgUnlabeld = os.path.join(inputPathImgUnlabeled, imgListUnlabeled[idxUnlabeled])
            while not fnCurrImgUnlabeld.endswith(".jpg") and not fnCurrImgUnlabeld.endswith(".png"):
                idxUnlabeled = np.random.randint(0, len(imgListUnlabeled))
                fnCurrImgUnlabeld = os.path.join(inputPathImgUnlabeled, imgListUnlabeled[idxUnlabeled])
            
            xSemi = cv.imread(fnCurrImgUnlabeld, cv.IMREAD_COLOR)
            xSemi = get_image_sized(xSemi, img_depth=3, out_size=in_out_size)           
                        
            images[b, :, :, :] = xSemi
            images[batchsize+b, :, :, :] = x            
            
            labelsSeg[b,:,:,0] = np.zeros(in_out_size, np.float32)            
            labelsSeg[batchsize+b,:,:,0] = yArea
            
            labelsOriMap[b,:,:,:] = np.zeros((in_out_size[0], in_out_size[1], 3), np.float32)
            labelsOriMap[batchsize+b,:,:,:] = yDirection
            
            labelsDistTrafo[b,:,:,0] = np.zeros(in_out_size, np.float32)            
            labelsDistTrafo[batchsize+b,:,:,0] = yTrafo
            
            labelsDistTrafoInv[b,:,:,0] = np.zeros(in_out_size, np.float32)            
            labelsDistTrafoInv[batchsize+b,:,:,0] = yTrafoInv

            labelsBound[b,:,:,0] = np.zeros(in_out_size, np.float32)
            labelsBound[batchsize + b, :, :, 0] = yBound

            b+=1
            if (b==batchsize):
                yield (images, [labelsOriMap, labelsDistTrafo, labelsDistTrafoInv, labelsBound, labelsSeg])
                b=0
                
# ====================================================================================================================================================
def get_image_sized(in_image, img_depth=1, out_size=(448,448)):
    # resizes the image 
    # out_size in order (columns, rows)
    x=cv.resize(in_image, (out_size[0], out_size[1]))
    x=x.astype('float32')
    x/=255.
    if (img_depth==1):
        x=np.expand_dims(x,axis=2)
    return x

# ====================================================================================================================
reduce_lr = K.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=15, min_lr=1e-09, verbose=1)

# ====================================================================================================================
checkpointer = K.callbacks.ModelCheckpoint('ConsInstancy_{epoch:03d}-{loss:.4f}.hdf5',
                                               verbose=0, 
                                               monitor='loss',
                                               save_best_only=True, 
                                               save_weights_only=True, 
                                               mode='auto',
                                               period=1)
                          