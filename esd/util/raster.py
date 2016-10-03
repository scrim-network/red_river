import numpy as np
from osgeo import gdal,osr
import pandas as pd

PROJECTION_GEO_WGS84 = 4326  # EPSG Code
PROJECTION_GEO_NAD83 = 4269  # EPSG Code
PROJECTION_GEO_WGS72 = 4322  # EPSG Code

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
    
class OutsideExtent(Exception):
    pass

class RasterDataset(object):
    '''
    Encapsulates a basic 2-D GDAL raster object with
    common utility functions.
    '''

    def __init__(self, ds_path):
        '''
        Parameters
        ----------
        ds_path : str
            A file path to a raster dataset (eg GeoTIFF file)
        '''

        self.gdal_ds = gdal.Open(ds_path)
        self.ds_path = ds_path
        # GDAL GeoTransform.
        # Top left x,y are for the upper left corner of upper left pixel
        # GeoTransform[0] /* top left x */
        # GeoTransform[1] /* w-e pixel resolution */
        # GeoTransform[2] /* rotation, 0 if image is "north up" */
        # GeoTransform[3] /* top left y */
        # GeoTransform[4] /* rotation, 0 if image is "north up" */
        # GeoTransform[5] /* n-s pixel resolution */
        self.geo_t = np.array(self.gdal_ds.GetGeoTransform())

        self.projection = self.gdal_ds.GetProjection()
        self.source_sr = osr.SpatialReference()
        self.source_sr.ImportFromWkt(self.projection)
        self.target_sr = osr.SpatialReference()
        self.target_sr.ImportFromEPSG(PROJECTION_GEO_WGS84)
        self.coordTrans_src_to_wgs84 = osr.CoordinateTransformation(self.source_sr, self.target_sr)
        self.coordTrans_wgs84_to_src = osr.CoordinateTransformation(self.target_sr, self.source_sr)

        self.min_x = self.geo_t[0]
        self.max_x = self.min_x + (self.gdal_ds.RasterXSize * self.geo_t[1])
        self.max_y = self.geo_t[3]
        self.min_y = self.max_y - (-self.gdal_ds.RasterYSize * self.geo_t[5])

        self.rows = self.gdal_ds.RasterYSize
        self.cols = self.gdal_ds.RasterXSize
        self.ndata = self.gdal_ds.GetRasterBand(1).GetNoDataValue()


    def get_coord_mesh_grid(self):
        '''
        Build a native projection coordinate mesh grid for the raster
        
        Returns
        ----------
        yGrid : ndarray
            A 2-D ndarray with the same shape as the raster containing
            the y-coordinate of the center of each grid cell.
        xGrid : ndarray
            A 2-D ndarray with the same shape as the raster containing
            the x-coordinate of the center of each grid cell.
        '''

        # Get the upper left and right point x coordinates in the raster's projection
        ulX = self.geo_t[0] + (self.geo_t[1] / 2.0)
        urX = self.get_coord(0, self.gdal_ds.RasterXSize - 1)[1]

        # Get the upper left and lower left y coordinates
        ulY = self.geo_t[3] + (self.geo_t[5] / 2.0)
        llY = self.get_coord(self.gdal_ds.RasterYSize - 1, 0)[0]

        # Build 1D arrays of x,y coords
        x = np.linspace(ulX, urX, self.gdal_ds.RasterXSize)
        y = np.linspace(ulY, llY, self.gdal_ds.RasterYSize)

        xGrid, yGrid = np.meshgrid(x, y)

        return yGrid, xGrid

    def get_coord_grid_1d(self):
        '''
        Build 1-D native projection coordinates for y and x dimensions
        of the raster.
        
        Returns
        ----------
        y : ndarray
            A 1-D array of the y-coordinates of each row.
        x : ndarray
            A 1-D array of the x-coordinates of each column.
        '''

        # Get the upper left and right point x coordinates in the raster's projection
        ulX = self.geo_t[0] + (self.geo_t[1] / 2.0)
        urX = self.get_coord(0, self.gdal_ds.RasterXSize - 1)[1]

        # Get the upper left and lower left y coordinates
        ulY = self.geo_t[3] + (self.geo_t[5] / 2.0)
        llY = self.get_coord(self.gdal_ds.RasterYSize - 1, 0)[0]

        # Build 1D arrays of x,y coords
        x = np.linspace(ulX, urX, self.gdal_ds.RasterXSize)
        y = np.linspace(ulY, llY, self.gdal_ds.RasterYSize)

        return y, x

    def get_coord(self, row, col):
        '''
        Get the native projection coordinates for a specific grid cell
        
        Parameters
        ----------
        row : int
            The row of the grid cell (zero-based)
        col : int
            The column of the grid cell (zero-based)
            
        Returns
        ----------
        yCoord : float
            The y projection coordinate of the grid cell.
        xCoord : float
            The x projection coordinate of the grid cell.   
        '''

        xCoord = (self.geo_t[0] + col * self.geo_t[1] + row * self.geo_t[2]) + self.geo_t[1] / 2.0
        yCoord = (self.geo_t[3] + col * self.geo_t[4] + row * self.geo_t[5]) + self.geo_t[5] / 2.0

        return yCoord, xCoord

    def get_row_col(self, lon, lat, check_bounds=True):
        '''
        Get the row, column grid cell offset for the raster based on an input
        WGS84 longitude, latitude point.
        
        Parameters
        ----------
        lon : float
            The longitude of the point
        lat : float
            The latitude of the point
        check_bounds : bool
            If True, will check if the point is within the bounds
            of the raster and raise a ValueError if not. If set to False and
            point is outside raster, the returned row, col will be clipped to
            the raster edge. 
        Returns
        ----------
        row : int
            The row of the closet grid cell to the lon,lat point (zero-based)
        col : int
            The column of the closet grid cell to the lon,lat point (zero-based) 
        '''

        xGeo, yGeo, zGeo = self.coordTrans_wgs84_to_src.TransformPoint(lon, lat)
        
        if check_bounds:

            if not self.is_inbounds(xGeo, yGeo):
                raise ValueError("lat/lon outside raster bounds: " + str(lat) + "," + str(lon))

        originX = self.geo_t[0]
        originY = self.geo_t[3]
        pixelWidth = self.geo_t[1]
        pixelHeight = self.geo_t[5]

        xOffset = np.abs(np.int((xGeo - originX) / pixelWidth))
        yOffset = np.abs(np.int((yGeo - originY) / pixelHeight))
        
        # clip row,col if outside raster bounds
        row = self._check_cellxy_valid(int(yOffset), self.rows)
        col = self._check_cellxy_valid(int(xOffset), self.cols)
            
        return row, col

    def get_data_value(self, lon, lat):
        '''
        Get the nearest grid cell data value to an input
        WGS84 longitude, latitude point. Will raise an 
        OutsideExtent exception if the longitude, latitude 
        is outside the bounds of the raster.
        
        Parameters
        ----------
        lon : float
            The longitude of the point
        lat : float
            The latitude of the point
            
        Returns
        ----------
        data_val : dtype of raster
            The data value of the closet grid cell to the lon,lat point         
        '''

        row, col = self.get_row_col(lon, lat)
        data_val = self.gdal_ds.ReadAsArray(col, row, 1, 1)[0, 0]

        return data_val

    def read_as_array(self):
        '''
        Read in the entire raster as a 2-D array
        
        Returns
        ----------
        a : MaskedArray
            A 2-D MaskedArray of the raster data. No data values
            are masked.      
        '''

        a = self.gdal_ds.GetRasterBand(1).ReadAsArray()
        a = np.ma.masked_equal(a, self.gdal_ds.GetRasterBand(1).GetNoDataValue())
        return a
    
    def output_new_ds(self, fpath, a, gdal_dtype, ndata=None, gdal_driver="GTiff"):
        '''
        Output a new raster with same geotransform and projection as this RasterDataset
        
        Parameters
        ----------
        fpath : str
            The filepath for the new raster.
        a : ndarray or MaskedArray
            A 2-D array of the raster data to be output.
            If MaskedArray, masked values are set to ndata
        gdal_dtype : str
            A gdal datatype from gdalconst.GDT_*.
        ndata : num, optional
            The no data value for the raster
        gdal_driver : str, optional
            The GDAL driver for the output raster data format
        '''
        
        ds_out = gdal.GetDriverByName(gdal_driver).Create(fpath, int(a.shape[1]), int(a.shape[0]), 1, gdal_dtype)
        ds_out.SetGeoTransform(self.gdal_ds.GetGeoTransform())
        ds_out.SetProjection(self.gdal_ds.GetProjection())
        
        band_out = ds_out.GetRasterBand(1)
        if ndata is not None:
            band_out.SetNoDataValue(ndata)
        band_out.WriteArray(np.ma.filled(a, ndata))
                    
        ds_out.FlushCache()
        ds_out = None

    def resample_to_ds(self, fpath, ds_src, gdal_gra, gdal_driver="GTiff"):
        '''
        Resample a different RasterDataset to this RasterDataset's grid
        
        Parameters
        ----------
        fpath : str
            The filepath for the new resampled raster.
        ds_src : RasterDataset
            The RasterDataset to  be resampled
        gdal_gra : str
            A gdal resampling algorithm from gdalconst.GRA_*.
        gdal_driver : str, optional
            The GDAL driver for the output raster data format
            
        Returns
        ----------
        grid_out : RasterDataset
            A RasterDataset pointing to the resampled raster  
        '''
        
        grid_src = ds_src.gdal_ds
        grid_dst = self.gdal_ds
        
        proj_src = grid_src.GetProjection()
        dtype_src = grid_src.GetRasterBand(1).DataType
        ndata_src = grid_src.GetRasterBand(1).GetNoDataValue()
        
        proj_dst = grid_dst.GetProjection()
        geot_dst = grid_dst.GetGeoTransform()
            
        grid_out = gdal.GetDriverByName(gdal_driver).Create(fpath, grid_dst.RasterXSize,
                                                        grid_dst.RasterYSize, 1, dtype_src)
        
        if ndata_src is not None:
            band = grid_out.GetRasterBand(1)
            band.Fill(ndata_src)
            band.SetNoDataValue(ndata_src)
        
        grid_out.SetGeoTransform(geot_dst)
        grid_out.SetProjection(proj_dst)
        grid_out.FlushCache()
        
        gdal.ReprojectImage(grid_src, grid_out, proj_src, proj_dst, gdal_gra)
        grid_out.FlushCache()
        # Make sure entire grid is written by setting to None. 
        # FlushCache doesn't seem to write the entire grid after resampling?
        grid_out = None
        
        # return as RasterDataset
        grid_out = RasterDataset(fpath)
        
        return grid_out
        
    def is_inbounds(self, x_geo, y_geo):
        return x_geo >= self.min_x and x_geo <= self.max_x and y_geo >= self.min_y and y_geo <= self.max_y
    
    def _check_cellxy_valid(self, i, n):
        
        if i < 0:
            i = 0
        elif i >= n:
            i = n - 1
        
        return i

def buffer_cdo_lonlat_grid_cfg(fpath_cdo_grid_cfg, buf, new_inc=None):
    '''
    Create a new CDO lonlat grid configuration based on applying a buffer to 
    an existing CDO grid configuration.
    
    Parameters
    ----------
    fpath_cdo_grid_cfg : str
        Path to existing CDO grid configuration
    buf : float
        Buffer in degrees
    new_inc : float, optional
        New grid increment/resolution
        
    Returns
    ----------
    lines_out : list of str
        List of lines for new CDO configuration file
    '''
    grid_cfg = pd.read_csv(fpath_cdo_grid_cfg, sep='=', index_col=0,
                           header=None, squeeze=True, skiprows=1)
    grid_cfg.index = grid_cfg.index.str.strip()
     
    xlast = grid_cfg.loc['xfirst'] + ((grid_cfg.loc['xsize']-1) * grid_cfg.loc['xinc'])
    lon = np.linspace(grid_cfg.loc['xfirst'], xlast, num=grid_cfg.loc['xsize'])
     
    ylast = grid_cfg.loc['yfirst'] + ((grid_cfg.loc['ysize']-1) * grid_cfg.loc['yinc'])
    lat = np.linspace(grid_cfg.loc['yfirst'], ylast, num=grid_cfg.loc['ysize'])
          
    buf_xfirst = lon[0] - buf
    buf_xlast = lon[-1] + buf
    buf_nx = (np.abs((lon[0] - buf) - (lon[-1] + buf))/(grid_cfg.loc['xinc'])) + 1
    buf_lon = np.linspace(buf_xfirst, buf_xlast, buf_nx)
     
    buf_yfirst = lat[0] - buf
    buf_ylast = lat[-1] + buf
    buf_ny = (np.abs((lat[0] - buf) - (lat[-1] + buf))/(grid_cfg.loc['yinc'])) + 1
    buf_lat = np.linspace(buf_yfirst, buf_ylast, buf_ny)
    
    if new_inc is None:
        
        xinc = grid_cfg.loc['xinc']
        yinc = grid_cfg.loc['yinc']
        
    else:
        
        x_extent_left =  buf_xfirst - (grid_cfg.loc['xinc']/2.0)
        y_extent_left =  buf_yfirst - (grid_cfg.loc['yinc']/2.0)
        
        x_extent_right =  buf_xlast + (grid_cfg.loc['xinc']/2.0)
        y_extent_right =  buf_ylast + (grid_cfg.loc['yinc']/2.0)
        
        buf_xfirst = x_extent_left + (new_inc/2.0)
        buf_xlast = x_extent_right - (new_inc/2.0)
        buf_nx = (np.abs(buf_xlast-buf_xfirst)/(new_inc)) + 1
        buf_lon = np.linspace(buf_xfirst, buf_xlast, buf_nx)
        
        buf_yfirst = y_extent_left + (new_inc/2.0)
        buf_ylast = y_extent_right - (new_inc/2.0)
        buf_ny = (np.abs(buf_ylast-buf_yfirst)/(new_inc)) + 1
        buf_lat = np.linspace(buf_yfirst, buf_ylast, buf_ny)
        
        xinc = new_inc
        yinc = new_inc
    
    lines_out = ['gridtype = lonlat\n']
    lines_out.append('xsize    = %d\n'%buf_nx)
    lines_out.append('ysize    = %d\n'%buf_ny)
    lines_out.append('xfirst   = %f\n'%buf_lon[0])
    lines_out.append('xinc     = %f\n'%xinc)
    lines_out.append('yfirst   = %f\n'%buf_lat[0])
    lines_out.append('yinc     = %f\n'%yinc)
     
    return lines_out

