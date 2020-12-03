import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image Handling
def load_images(path, mode='color'):
    # load all images in path
    """
   FUNCTION: load_images
        Call to load images colored or grayscale and stack them.
     INPUTS:
        path = location of image
        mode = 'grayscale' or 'colored'
    OUTPUTS:
        read data file
    """
    image_stack = []
    for i, filename in enumerate(os.listdir(path)):
        print("Loading... /" + filename + "...as Image_stack["+str(i)+"]")
        if mode == 'color':
            image = cv2.imread(os.path.join(path, filename), cv2.IMREAD_COLOR)
        else: #mode == 'gray':
            image = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        image_stack.append(image)
    print("\n")
    return image_stack

def alignment(image_stack):
    # resize a list images to the smallest one
    """
   FUNCTION: alignmentent
        Call to Create Uniform Images by adjusting image sizes
     INPUTS:
        image_stack = stack of images from load_images
    OUTPUTS:
        images files of the same size
    """
    sizes = [img.shape for img in image_stack] # e.g., (D, h, w, c)
    minh, minw, _ = np.min(sizes, axis=0)

    for i, img in enumerate(image_stack):
        if np.shape(img)[:2] !=  (minh, minw):
            print("Warning: Detected Non-Constant Sized Image"+str(i)+"of size "+str(sizes[i]))
            image_stack[i] = cv2.resize(img, (minw, minh))
    return image_stack

# Quality Measures
def contrast(image, ksize=1):
    """
   FUNCTION: contrast
        Call to compute the first quality measure: contrast using laplacian kernel
     INPUTS:
        image = input image (colored)
        ksize = 1 means: [[0,1,0],[1,-4,1],[0,1,0]] kernel
    OUTPUTS:
        contrast measure
    """
    image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2GRAY) # gray scale
    laplacian = cv2.Laplacian(image.astype('float64'), cv2.CV_64F, ksize)
    C = cv2.convertScaleAbs(laplacian)
    C = cv2.medianBlur(C.astype('float32') , 5)
    return C.astype('float64')

def saturation(image):
    """
   FUNCTION: saturation
        Call to compute second quality measure - st.dev across RGB channels
     INPUTS:
        image = input image (colored)
    OUTPUTS:
        saturation measure
    """
    # gray scale
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    S = np.std(image, 2)
    return S.astype('float64')

def exposedness(image, sigma=0.2):
    """
   FUNCTION: exposedness
        Call to compute third quality measure - exposure using a gaussian curve
     INPUTS:
        image = input image (colored)
        sigma = gaussian curve parameter
    OUTPUTS:
        exposedness measure
    """
    #image = cv2.normalize(image, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    image = image/255. # FIXME
    gauss_curve = lambda i : np.exp(-((i-0.5)**2) / (2*sigma*sigma))
    R_gauss_curve = gauss_curve(image[:,:,2])
    G_gauss_curve = gauss_curve(image[:,:,1])
    B_gauss_curve = gauss_curve(image[:,:,0])
    E = R_gauss_curve * G_gauss_curve * B_gauss_curve
    return E.astype('float64')

def scalar_weight_map(image_stack, weights=[1,1,1]):
    """
   FUNCTION: scalar_weight_map
        Call to forcefully "AND"-combine all quality measures defined
     INPUTS:
        image_measures = stack of quality measures computed for image i
        image_measures[contrast, saturation, exposedness]
        weights = weight for each quality measure : weights[wc, ws, we]
    OUTPUTS:
        scalar_weight_map for particular image of shape (N, H, W)
    """
    D, H, W, _ = np.shape(image_stack)
    Wijk = np.zeros((D,H,W), dtype='float64')
    wc, ws, we = weights
    print("Computing Weight Maps from Measures using: C=%1.1d, S=%1.1d, E=%1.1d" %(wc,ws,we))

    # calculate pixel-wise weight map
    epsilon = 5e-6
    for i, img in enumerate(image_stack):
        C  = contrast(img) # laplacian filter to grayscale image
        S  = saturation(img) # standar deviate among rgb
        E  = exposedness(img) # closer to 0.5
        Wijk[i] = (np.power(C,wc)*np.power(S,ws)*np.power(E,we)) + epsilon # pixel-wise

    # normalize
    normalizer = np.sum(Wijk,0) # (428, 642) pixel-wise normalizer
    Wijk = Wijk / normalizer.reshape((1,H,W))

    print(" *Done");print("\n")

    return Wijk.astype('float64')

# Fusions
def measures_fusion_naive(image_stack, weight_maps, blurType = None, blurSize = (0,0), blurSigma = 15):
    """
   FUNCTION: measures_fusion_naive
        Call to fuse normalized weightmaps and their images
    INPUTS:
        image_stack = list contains the stack of "exposure-bracketed" images
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        weight_maps = scalar_weight_map for N images
        blurType    = gaussian or bilateral filter applied to weight-map
        blurSize/Sigma = blurring parameters
    OUTPUTS:
        img_fused = single image with fusion of measures
        Rij = fusion of individual images with their weight maps
    """
    D, H, W, _ = np.shape(image_stack)
    img_fused = np.zeros((H,W,3), dtype='float64')

    Rij  = []
    for i in range(D):
        if blurType == None:
            if i == 0: print("Performing Naive Blending")
            weight_map = weight_maps[i]
        elif blurType == 'gaussian':
            if i == 0: print("Performing Gaussian-Blur Blending")
            weight_map = cv2.GaussianBlur(weight_maps[i], blurSize, blurSigma)
        elif blurType == 'bilateral':
            if i == 0: print("Performing Bilateral-Blur Blending")
            weight_map = cv2.bilateralFilter(weight_maps[i].astype('float32'), blurSigma, blurSize[0], blurSize[1])
        Rijk = image_stack[i]* np.dstack([weight_map,weight_map,weight_map])
        Rij.append(Rijk)
        img_fused += Rijk

    print(" *Done");print("\n")

    return img_fused, Rij

def multires_pyramid(image, weight_map, levels):
    """
   FUNCTION: multires_pyramid
        Call to compute image and weights pyramids
    INPUTS:
        image = numpy array image
        weight_map = scalar_weight_map for N images
        levels = height of pyramid to use including base pyramid base
    OUTPUTS:
        imgLpyr = list containing image laplacian pyramid
        wGpyr   = list containing weight gaussian pyramid
    """
    levels  = levels - 1

    # Gaussian pyramid
    imgGpyr = [image] # image Gaussian pyramid
    wGpyr   = [weight_map] # weight Gaussian pyramid

    for i in range(levels):
        imgGpyr.append(cv2.pyrDown(imgGpyr[i].astype('float64')))
        wGpyr.append(cv2.pyrDown(wGpyr[i].astype('float64')))

    # Laplacian pyramid
    imgLpyr = [imgGpyr[levels]]
    wLpyr = [wGpyr[levels]]

    for i in range(levels, 0, -1):
        H, W, _ = np.shape(imgGpyr[i-1])
        imgLpyr.append(imgGpyr[i-1] - cv2.resize(cv2.pyrUp(imgGpyr[i]),(W,H)))
        wLpyr.append(wGpyr[i-1] - cv2.resize(cv2.pyrUp(wGpyr[i]),(W,H)))

    imgLpyr = imgLpyr[::-1]
    wLpyr = wLpyr[::-1]

    return imgLpyr, wGpyr


def measures_fusion_multires(image_stack, weight_maps, levels=6):
    """
   FUNCTION: measures_fusion_multires
        Call to perform multiresolution blending
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        levels = desired height of the pyramids
        weight_maps = scalar_weight_map for N images
    OUTPUTS:
        finalImage = single exposure fused image
    """
    print("Performing Multiresolution Blending using: "+str(levels)+" Pyramid levels")

    D = len(image_stack)

    # get weighted pyramids for each image
    weightedPyramids = [] # weightedPyramids[i][j] -> i-th image j-th level pyramid
    for i in range(D): # image-wise
        imgLpyr, wGpyr = multires_pyramid(image_stack[i].astype('float64'), weight_maps[i], levels)
        blended_multires = []
        for j in range(levels): # level-wise
            blended_multires.append(imgLpyr[j] * np.expand_dims(wGpyr[j], axis=-1))
        weightedPyramids.append(blended_multires)

    # combine weighted pyramids among images
    import ipdb; ipdb.set_trace()  # XXX DEBUG

    weightedsumPyramid = []
    for i in range(levels):
        # initialize with 0
        tmp = np.zeros_like(weightedPyramids[0][i]) # first image, i-th level
        # sum weighted inputs over images
        for j in range(D):
            tmp += np.array(weightedPyramids[j][i])
        weightedsumPyramid.append(tmp)

    ## reconstruct a single image from a (blended) laplacian pyramid
    weightedsumPyramid = weightedsumPyramid[::-1]
    layerx = weightedsumPyramid[0]
    for i in range(1, levels):
        H, W, _ = weightedsumPyramid[i].shape
        layerx = weightedsumPyramid[i] + cv2.resize(cv2.pyrUp(layerx),(W,H))

    imgH, imgW, C = image_stack[0].shape
    finalImage = cv2.resize(layerx, (imgW, imgH))

    # clipping for overflow
    finalImage[finalImage < 0] = 0
    finalImage[finalImage > 255] = 255
    print(" *Done"); print("\n")

    return finalImage.astype('uint8')


def measures_iterative_fusion_multires(image_stack, levels=6):
    levels = levels - 1
    # GP & LP
    imgGpyrs = []; imgLpyrs = []
    for image in image_stack:
        image = image.astype('float64')
        # Gaussian pyramid
        imgGpyr = [image]
        for i in range(levels):
            imgGpyr.append(cv2.pyrDown(imgGpyr[i].astype('float32')))
        imgGpyrs.append(imgGpyr)

        # Laplacian pyramid
        imgLpyr = [imgGpyr[levels]]
        for i in range(levels, 0, -1):
            H, W, _ = np.shape(imgGpyr[i-1])
            imgLpyr.append(imgGpyr[i-1] - cv2.resize(cv2.pyrUp(imgGpyr[i]),(W,H)))
        imgLpyr = imgLpyr[::-1]
        imgLpyrs.append(imgLpyr)

    # weighted pyramids
    weightedsumPyramid = []
    for i in range(levels+1):
        imgGlvl_stack = np.array([imgGpyr[i] for imgGpyr in imgGpyrs])
        imgLlvl_stack = np.array([imgLpyr[i] for imgLpyr in imgLpyrs])
        weight_maps = scalar_weight_map(imgGlvl_stack, weights=[1,1,1])
        weighted_lvl = imgLlvl_stack * np.expand_dims(weight_maps, axis=-1)
        weightedsum_lvl = np.sum(weighted_lvl, 0)
        weightedsumPyramid.append(weightedsum_lvl)

    ## reconstruct a single image from a (blended) laplacian pyramid
    weightedsumPyramid = weightedsumPyramid[::-1]
    layerx = weightedsumPyramid[0]
    for i in range(1, levels+1):
        H, W, _ = weightedsumPyramid[i].shape
        layerx = weightedsumPyramid[i] + cv2.resize(cv2.pyrUp(layerx),(W,H))

    imgH, imgW, C = image_stack[0].shape
    finalImage = cv2.resize(layerx, (imgW, imgH))

    # clipping for overflow
    print(finalImage.min(), finalImage.max())
    finalImage[finalImage < 0] = 0
    finalImage[finalImage > 255] = 255
    print(" *Done"); print("\n")

    return finalImage.astype('uint8')

# FIXME: WHAT ARE THOSE?
###############################################################################
"""
Compute Mean of Image Stack
"""
###############################################################################
def meanImage(image_stack, save=False):
    """
   FUNCTION: meanImage
        Call to perform mean image blending
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        save = save figures to directory
    OUTPUTS:
        mean of all the images in the stack
    """
#'-----------------------------------------------------------------------------#
    N = len(image_stack)
    H = np.shape(image_stack[0])[0]
    W = np.shape(image_stack[0])[1]
    rr = np.zeros((H,W), dtype='float64')
    gg = np.zeros((H,W), dtype='float64')
    bb = np.zeros((H,W), dtype='float64')
    for i in range(N):
        r, g, b = cv2.split(image_stack[i].astype('float64'))
        rr += r.astype('float64')
        gg += g.astype('float64')
        bb += b.astype('float64')
    MeanImage = np.dstack([rr/N,gg/N,bb/N]).astype('uint8')
    if save == True:
        cv2.imwrite('img_MeanImage.png', MeanImage)
    return MeanImage
###############################################################################





"""
Visualize Image Measures, Weight Maps
"""
###############################################################################
def visualize_maps(image_stack, weights=[1,1,1], save=False):
    """
   FUNCTION: measures_fusion_multires
        Call to perform multiresolution blending
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        weights = importance factor for each measure C,S,E
        save = save figures to directory
    OUTPUTS:
        images of contrast, saturation, exposure, and combined weight for image N
    """
#'-----------------------------------------------------------------------------#
    for N in range(len(image_stack)):
        img_contrast    = contrast(image_stack[N])
        img_saturation  = saturation(image_stack[N])
        img_exposedness = exposedness(image_stack[N])
        #weight_map      = scalar_weight_map([image_stack[N]], weights)
        print("Displaying Measures and Weight Map for Image_stack["+str(N)+"]")

        if save == False:
            plt.figure(1);plt.imshow(img_contrast.astype('float'),cmap='gray')
            plt.figure(2);plt.imshow(img_saturation,cmap='gray')
            plt.figure(3);plt.imshow(img_exposedness,cmap='gray')
            #plt.figure(4);plt.imshow(weight_map[:,:,0],cmap='gray') #.astype('uint8')
        else:
            plt.imsave('img_contrast'+str(N)+'.png', img_contrast, cmap = 'gray', dpi=1800)
            plt.imsave('img_saturation'+str(N)+'.png', img_saturation, cmap = 'gray', dpi=1800)
            plt.imsave('img_exposedness'+str(N)+'.png', img_exposedness, cmap = 'gray', dpi=1800)
            #weight_map = 255*cv2.normalize(weight_map[:,:,0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
            #plt.imsave('weightmaps_combined_Normalized'+str(N)+'.png', weight_map.astype('uint8'), cmap = 'gray', dpi=1800)
    print(" *Done"); print("\n")
###############################################################################



