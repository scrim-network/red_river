import os
import errno
import time
import sys
import numpy as np

def mkdir_p(path):
    '''
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise 
        
class StatusCheck(object):
    '''
    Class for printing out progress messages
    '''


    def __init__(self, total_cnt, check_cnt):
        '''
        total_cnt: the total number of items being processed
        check_cnt: the number of items completed at which a progress message
        should be printed
        '''
        self.total_cnt = total_cnt
        self.check_cnt = check_cnt
        self.num = 0 
        self.num_last_check = 0
        self.status_time = time.time()
        self.start_time = self.status_time
    
    def increment(self, n=1):
        
        self.num += n
        
        if self.num - self.num_last_check >= self.check_cnt:
            
            currentTime = time.time()
            
            if self.total_cnt != -1:
                
                print ("Total items processed is %d.  Last %d items took "
                       "%f minutes. %d items to go." % 
                       (self.num, self.num - self.num_last_check,
                        (currentTime - self.status_time) / 60.0,
                        self.total_cnt - self.num))
                print ("Current total process time: %f minutes" %
                       ((currentTime - self.start_time) / 60.0))
                print ("Estimated Time Remaining: %f" %
                       (((self.total_cnt - self.num) / float(self.num)) * 
                        ((currentTime - self.start_time) / 60.0)))
            
            else:
            
                print ("Total items processed is %d.  "
                       "Last %d items took %f minutes" %
                       (self.num, self.num - self.num_last_check,
                        (currentTime - self.status_time) / 60.0))
                print ("Current total process time: %f minutes" %
                       ((currentTime - self.start_time) / 60.0))
            
            sys.stdout.flush()
            self.status_time = time.time()
            self.num_last_check = self.num
            
RADIAN_CONVERSION_FACTOR = 0.017453292519943295 #pi/180
AVG_EARTH_RADIUS_KM = 6371.009 #Mean earth radius as defined by IUGG

def grt_circle_dist(lon1,lat1,lon2,lat2):
    '''
    Calculate great circle distance according to the haversine formula
    see http://en.wikipedia.org/wiki/Great-circle_distance
    '''
    #convert to radians
    lat1rad = lat1 * RADIAN_CONVERSION_FACTOR
    lat2rad = lat2 * RADIAN_CONVERSION_FACTOR
    lon1rad = lon1 * RADIAN_CONVERSION_FACTOR
    lon2rad = lon2 * RADIAN_CONVERSION_FACTOR
    deltaLat = lat1rad - lat2rad
    deltaLon = lon1rad - lon2rad
    centralangle = 2 * np.arcsin(np.sqrt((np.sin (deltaLat/2))**2 + np.cos(lat1rad) * np.cos(lat2rad) * (np.sin(deltaLon/2))**2))
    #average radius of earth times central angle, result in kilometers
    #distDeg = centralangle/RADIAN_CONVERSION_FACTOR
    distKm = AVG_EARTH_RADIUS_KM * centralangle 
    return distKm
    
def runs_of_ones_array(bits):
    # http://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_ends - run_starts