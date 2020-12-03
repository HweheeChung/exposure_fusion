"""
MAIN FILE :: EXPOSURE FUSION

Abdulmajeed Muhammad Kabir

This is the main file to run Exposure fusion using exposureFusion.py as library
"""
import exposureFusion_ as ef



"Load Input Images and Setup Dirs"
path = './inputs'
resultsPath = './results'

image_stack  = ef.load_images(path) # read images on paths to a list format
image_stack  = ef.alignment(image_stack) # resize images to min(w,h) among image list

if not ef.os.path.exists(resultsPath):
    ef.os.mkdir(resultsPath)


"Multiresolution Iterative Exposure Fusion"
#------------------------------------------------------------------------------#
final_imageD = ef.measures_iterative_fusion_multires(image_stack, levels=6)
#ef.cv2.imshow('Iterative Multiresolution Fusion', final_imageD.astype('uint8'))
##ef.cv2.waitKey(0); ef.cv2.destroyAllWindows()
ef.cv2.imwrite(ef.os.path.join(resultsPath,'img_IterativeMultiresolutionFusion.png'), final_imageD.astype('uint8'))



"Compute Quality Measures"
#------------------------------------------------------------------------------#
#Compute Quality measures multiplied and weighted with weights[x,y,z]
weight_map = ef.scalar_weight_map(image_stack, weights = [1,1,1])
#weight_map = ef.scalar_weight_map(image_stack, weights = [0,0,0]) #Performs Pyramid Fusion




"Original Image"
#------------------------------------------------------------------------------#
#load original image i.e center image probably has the median Exposure value(EV)
filename = sorted(ef.os.listdir(path))[len(ef.os.listdir(path))//2]
original_image = ef.cv2.imread(ef.os.path.join(path, filename), ef.cv2.IMREAD_COLOR)
#ef.cv2.imshow('Original Image', original_image)
##ef.cv2.waitKey(0); ef.cv2.destroyAllWindows()
ef.cv2.imwrite(ef.os.path.join(resultsPath,'img_CenterOriginal.png'), original_image.astype('uint8'))





"Naive Exposure Fusion"
#------------------------------------------------------------------------------#
final_imageA, RijA = ef.measures_fusion_naive(image_stack, weight_map)
#ef.cv2.imshow('Naive Fusion', final_imageA.astype('uint8'))
##ef.cv2.waitKey(0); ef.cv2.destroyAllWindows()
ef.cv2.imwrite(ef.os.path.join(resultsPath,'img_NaiveFusion.png'), final_imageA.astype('uint8'))





"Blurred Exposure Fusion"
#------------------------------------------------------------------------------#
final_imageB, RijB = ef.measures_fusion_naive(image_stack, weight_map, blurType = 'gaussian', blurSize = (0,0), blurSigma = 15)
#ef.cv2.imshow('Blur Fusion', final_imageB.astype('uint8'))
##ef.cv2.waitKey(0); ef.cv2.destroyAllWindows()
ef.cv2.imwrite(ef.os.path.join(resultsPath,'img_BlurFusion.png'), final_imageB.astype('uint8'))





"Bilateral Exposure Fusion"
#------------------------------------------------------------------------------#
final_imageC, RijC = ef.measures_fusion_naive(image_stack, weight_map, blurType = 'bilateral', blurSize = (115,115), blurSigma = 51)
#ef.cv2.imshow('Bilateral Fusion', final_imageC.astype('uint8'))
##ef.cv2.waitKey(0); ef.cv2.destroyAllWindows()
ef.cv2.imwrite(ef.os.path.join(resultsPath,'img_BilateralFusion.png'), final_imageC.astype('uint8'))





"Multiresolution Exposure Fusion"
#------------------------------------------------------------------------------#
final_imageD = ef.measures_fusion_multires(image_stack, weight_map, levels=6)
#ef.cv2.imshow('Multiresolution Fusion', final_imageD.astype('uint8'))
##ef.cv2.waitKey(0); ef.cv2.destroyAllWindows()
ef.cv2.imwrite(ef.os.path.join(resultsPath,'img_MultiresolutionFusion.png'), final_imageD.astype('uint8'))





"Display Intermediate Steps and Save"
#------------------------------------------------------------------------------#
#ef.visualize_maps(image_stack, save=False)



"Compute Mean of Image Stack"
#------------------------------------------------------------------------------#
#final_imageE = ef.meanImage(image_stack, save=False)
