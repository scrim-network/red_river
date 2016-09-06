import xarray as xr
import numpy as np
from osgeo import gdal, gdalconst, osr
from twx.raster.raster_dataset import RasterDataset


BOUNDS_RED_RIVER_APHRODITE = 99.875,19.875,106.625,25.625

def grid_wgs84_to_raster(fpath, a, lon, lat, gdal_dtype, ndata=None, gdal_driver="GTiff"):
    
    '''
    Create GDAL geotransform list to define resolution and bounds
    GeoTransform[0] /* top left x */
    GeoTransform[1] /* w-e pixel resolution */
    GeoTransform[2] /* rotation, 0 if image is "north up" */
    GeoTransform[3] /* top left y */
    GeoTransform[4] /* rotation, 0 if image is "north up" */
    GeoTransform[5] /* n-s pixel resolution */
    '''
    geo_t = [None]*6
    #n-s pixel height/resolution needs to be negative.  not sure why?
    geo_t[5] = -np.abs(lat[0] - lat[1])   
    geo_t[1] = np.abs(lon[0] - lon[1])
    geo_t[2],geo_t[4] = (0.0,0.0)
    geo_t[0] = lon[0] - (geo_t[1]/2.0) 
    geo_t[3] = lat[0] + np.abs(geo_t[5]/2.0)
    
    ds_out = gdal.GetDriverByName(gdal_driver).Create(fpath, int(a.shape[1]),
                                                      int(a.shape[0]),
                                                      1, gdal_dtype)
    ds_out.SetGeoTransform(geo_t)
    
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    ds_out.SetProjection(sr.ExportToWkt())
    
    band_out = ds_out.GetRasterBand(1)
    if ndata is not None:
        band_out.SetNoDataValue(ndata)
    band_out.WriteArray(np.ma.filled(a, ndata))
                
    ds_out.FlushCache()
    ds_out = None

if __name__ == '__main__':
    
    bnds = BOUNDS_RED_RIVER_APHRODITE
    
    ds = xr.open_dataset('/storage/home/jwo118/group_store/red_river/aphrodite/APHRODITE.Monsoon_Asia.PCP.0p25deg.v1101r2.1961-2007.nc')
    
    mask_lat = (ds.lat >= bnds[1]) & (ds.lat <= bnds[3]) 
    mask_lon = (ds.lon >= bnds[0]) & (ds.lon <= bnds[2])
    
    da = ds.PCP[:,mask_lat,mask_lon].load()
    
    da = da.mean(dim='time')
    a = da.values
    a = np.flipud(a)
    a = np.ma.masked_invalid(a)
    
    lon = da.lon.values
    lat = da.lat.values
    lat = lat[::-1]
    
    grid_wgs84_to_raster('/storage/home/jwo118/group_store/red_river/aphrodite/redriver_bnds.tif',
                         a, lon, lat, gdalconst.GDT_Float32, ndata=-9999)
    
    ds = RasterDataset('/storage/home/jwo118/group_store/red_river/aphrodite/redriver_bnds.tif')
    
    # Get the upper left and right point x coordinates in the raster's projection
    ulX = ds.geo_t[0] + (ds.geo_t[1] / 2.0)
    urX = ds.get_coord(0, ds.gdal_ds.RasterXSize - 1)[1]

    ulx = ds.geo_t[0]
    uly = ds.geo_t[3]
    
    ptx = ulx + .5
    pty = uly - .5
    
    lat = np.linspace(pty, pty - 5, num=6)
    lon = np.linspace(ptx, ptx + 6, num=7)
    
    a = np.random.rand(lat.size, lon.size)
    
    grid_wgs84_to_raster('/storage/home/jwo118/group_store/red_river/aphrodite/redriver_1degbnds.tif',
                         a, lon, lat, gdalconst.GDT_Float32, ndata=-9999)
    
    ds = RasterDataset('/storage/home/jwo118/group_store/red_river/aphrodite/redriver_1degbnds.tif')
    
    
    
    
    