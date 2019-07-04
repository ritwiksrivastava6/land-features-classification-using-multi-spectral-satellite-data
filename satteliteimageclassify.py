import numpy as np
from osgeo import gdal, gdal_array,ogr
import matplotlib.pyplot as plt
import cv2

gdal.UseExceptions()
gdal.AllRegister()
#importing dataset from a shape file containing all land features like vegetation,urban area,mountains etc.
dataset = ogr.Open('D:\\sattelite data\\shappe2.shp')
if not dataset:
    print('Error: could not open dataset')
layer = dataset.GetLayerByIndex(0)
#getting layer count
layer_count = dataset.GetLayerCount()
print('The shapefile has {n} layer(s)\n'.format(n=layer_count))
#obtaining the spatial reference of the layers and exporting it to proj4
spatial_ref = layer.GetSpatialRef()
proj4 = spatial_ref.ExportToProj4()

print('Layer projection is: {proj4}\n'.format(proj4=proj4))
#counting the number of features in layer
feature_count = layer.GetFeatureCount()
print('Layer has {n} features\n'.format(n=feature_count))

#importing raster input image
img_ds = gdal.Open("D:\\sattelite data\\miscoutput\\img1.tif", gdal.GA_ReadOnly)


#obtaining the projection reference of the input raster image
proj = img_ds.GetProjectionRef()

ext = img_ds.GetGeoTransform()
print(ext)
print(proj)
ncol = img_ds.RasterXSize
nrow = img_ds.RasterYSize
raster_ds = None
memory_driver = gdal.GetDriverByName('GTiff')
# creating output .tif with same nuber of columns and rows as input raster image
out_raster_ds = memory_driver.Create('D:\\sattelite data\\shapefile1.tif', ncol, nrow, 1, gdal.GDT_Byte)
#setting the projection of the new tif file same as the input raster image
out_raster_ds.SetProjection(proj)
out_raster_ds.SetGeoTransform(ext)
b = out_raster_ds.GetRasterBand(1)
b.Fill(0)
# rasterizing the  land features dataset into the output raster image
status = gdal.RasterizeLayer(out_raster_ds,
                             [1],
                             layer,
                             None, None,
                             [0],
                             ['ALL_TOUCHED=TRUE',
                              'ATTRIBUTE=id']
                             )

# Close dataset
out_raster_ds = None

if status != 0:
    print("I don't think it worked...")
else:
    print("Success")
#Now importing both raster image dataset and output raster image again
img_ds = gdal.Open("D:\\sattelite data\\miscoutput\\img1.tif", gdal.GA_ReadOnly)

roi_ds = gdal.Open('D:\\sattelite data\\shapefile1.tif', gdal.GA_ReadOnly)

img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

print(img_ds.RasterCount)
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

plt.subplot(121)
plt.imshow(img[:, :, 2], cmap=plt.cm.Greys_r)
plt.title('raster image input')

plt.subplot(122)
plt.imshow(roi, cmap=plt.cm.Spectral)
plt.title('ROI Training Data')

plt.show()
#obtaining samples from the output raster image which are not NULL values
n_samples = (roi > 0).sum()
print('We have {n} samples'.format(n=n_samples))
# obtaining different classes of the datasets like vegetation etc
labels = np.unique(roi[roi>0])
print('the training data includes {n} classes :{classes}'.format(n=labels.size,
                                                                classes=labels))
X = img[roi>0, :]
y = roi[roi>0]
print('our matrix is sized :: {sz}'.format(sz=X.shape))
print('Our y array is sized: {sz}'.format(sz=y.shape))




#importing randomforest classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, oob_score=True)
rf = rf.fit(X, y)
print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))

bands = [1, 2, 3, 4]
# printing importance of each bands
for b, imp in zip(bands, rf.feature_importances_):
    print('Band {b} importance: {imp}'.format(b=b, imp=imp))


import pandas as pd
#creating a panda dataframe just like confusion matrix
df = pd.DataFrame()
df['truth'] = y
df['predict'] = rf.predict(X)

print(pd.crosstab(df['truth'], df['predict'], margins=True))
#predicting the rest of the image
new_shape = (img.shape[0] * img.shape[1], img.shape[2])
img_as_array = img[:,:,:4].reshape(new_shape)
print('reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

class_prediction = rf.predict(img_as_array)
class_prediction = class_prediction.reshape(img[:, :, 0].shape)
# cv2.imwrite("D:\\sattelite data\\imgarr.tif", img_as_array)
# cv2.imwrite("D:\\sattelite data\\classpred.tif", class_prediction)
# color scaling
def color_stretch(image, index, minmax=(0, 10000)):
    colors = image[:, :, index].astype(np.float64)

    max_val = minmax[1]
    min_val = minmax[0]

    colors[colors[:, :, :] > max_val] = max_val
    colors[colors[:, :, :] < min_val] = min_val

    for b in range(colors.shape[2]):
        colors[:, :, b] = colors[:, :, b] * 1 / (max_val - min_val)

    return colors

img543 = color_stretch(img, [2, 1, 0], (0, 10000))

n = class_prediction.max()
# creating a dictionary that will help assigning colors to different landfeatures
colors = dict((
    (0, (0, 0, 0)),  # Nodata
    (1, (0, 150, 0)),  # vegetatoin

    (3, (0, 0, 255)),  # Water
    (5, (160, 82, 45)),# Barren
    (2, (255, 0, 0)),#urban
    (4, (151 ,124, 83))# mountain
))
for k in colors:
    v = colors[k]
    _v = [_v / 255.0 for _v in v]
    colors[k] = _v
index_colors = [colors[key] if key in colors else
                    (1, 1, 1) for key in range(1, n + 1)]
cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', n)

plt.subplot(121)
plt.imshow(img[:,:,2])
# plotting the prediction
plt.subplot(122)
plt.imshow(class_prediction, cmap=cmap, interpolation='none')
plt.title("classified image")

plt.show()