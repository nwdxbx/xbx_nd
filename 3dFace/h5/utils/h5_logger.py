#coding: utf-8
import os
import sys
import logging

class Logger:
    def __init__(self,port,name="h5_spoof",logdir="/data/logs",debug=False):    
        log_dir = logdir
        log_file_prefix = name
        self.port = str(port)
        self.service_name = name + "_service"
        filename = log_file_prefix + "_" + self.port + ".log"
        self.filepath = os.path.join(log_dir,filename)

        self.logger_obj = logging.getLogger(self.service_name)
        self.logger_obj.setLevel(logging.DEBUG)
        formater = logging.Formatter("%(asctime)s|%(service_name)s|%(port_id)s|%(request_id)s|%(levelname)s|%(stage_name)s|%(message)s")

        fh = logging.FileHandler(self.filepath)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formater)    
        self.logger_obj.addHandler(fh)


    def writelog(self,stage_name,context,level="INFO",request_id="-001"):
        extra_info = {"service_name": self.service_name,
                 "request_id":   request_id,
                 "stage_name":   stage_name,
                 "port_id":      self.port}
        if level == "INFO":
            self.logger_obj.info(context,extra=extra_info)
        elif level == "WARNING":
            self.logger_obj.warning(context,extra=extra_info)
        elif level == "DEBUG":
            self.logger_obj.debug(context,extra=extra_info)
        elif level == "ERROR":
            self.logger_obj.error(context,extra=extra_info)
        elif level == "CRITICAL":
            self.logger_obj.critical(context,extra=extra_info)

"""
if __name__ == "__main__":
    log = Logger(8080,logdir="./",debug=False)
    log.writelog("img",'{"flag": "000"}',request_id="1123588")
"""
        