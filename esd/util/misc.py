import os
import errno
import time
import sys

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