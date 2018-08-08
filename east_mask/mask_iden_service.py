#!/usr/bin/env python  
#encoding: utf-8  
#Author:xxx
#Date:2017.04.01
#tips:
#if the socket was locked, then do the below command to release the socked file:
#rm -rf /tmp/net_check_service.sock
import atexit, os, sys, time, signal
from deamon import cdaemon
import socket
import json
import base64
import cv2
import redis
import cPickle as pickle
import threading,getopt,string
import text_mask
import struct
import numpy as np

IDEN_IMAGE_MAXIMUM_LENGTH=2*1024*1024# max image size is 2M

class ClientDaemon(cdaemon):
    def __init__(self, name,gpu,save_path, stdin=os.devnull, stdout=os.devnull, stderr=os.devnull, home_dir='.', umask=022, verbose=1):  
        cdaemon.__init__(self, save_path, stdin, stdout, stderr, home_dir, umask, verbose)  
        self.name = name
        #self.sock = sock
        self.gpu = gpu

    def recv(self,connection):
        """
        two stage receive
        """

        data = connection.recv(4)
        fileLen = struct.unpack('i',data)[0]
        code = '000'
        if fileLen > IDEN_IMAGE_MAXIMUM_LENGTH:
            code = '004'
        data    = ''
        while fileLen>0:
            readLen = 1024
            tmp = connection.recv(readLen)
            data    += tmp
            fileLen = fileLen - len(tmp)
        return code,data

    # def process(self,sess,net,connection):
    #     code,data    = self.recv(connection)
    #     data = json.loads(data)
    #     app_data    = json.loads(data['data'])
    #     request_id  = app_data['request_id']
    #     image_io    = app_data['img']
    #     try:
    #         buf = base64.b64decode(image_io)
    #         buf = np.asarray(bytearray(buf),dtype='uint8')
    #         image   = cv2.imdecode(buf,cv2.IMREAD_COLOR)

    #         if code == '000':
    #             ret_v   = iden.check_all(sess,net,image)
    #             bimg    = cv2.imencode('.jpg',ret_v)[1]
    #             base64_data = str(base64.b64encode(bimg))
    #             return request_id,'000',base64_data
    #         else:
    #             return request_id,code,''
    #     except TypeError as e: 
    #         print ('TypeError',e)
    #         image = None
    #         return request_id,'002',''
            

    def run(self, output_fn, **kwargs):
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock="/tmp/%s.sock"%(self.name)
        if os.path.exists(sock):
            os.unlink(sock)
        server.bind(sock)
        server.listen(20)
        maskalgo = text_mask.textDetectMask(self.gpu)
        while True:
            connection, address = server.accept()
            try:
                connection.settimeout(20000)
                code,data = self.recv(connection)
                request_id,code,base64_data = maskalgo.get_result(code,data)
                a={'request_id':request_id,'img':base64_data ,'code':code,'message':''}
                result = {'data':a,'ip_port':''}
                connection.send(json.dumps(result))          
            except socket.timeout:
                a={'request_id':request_id,'img':'' ,'code':'501','message':'socket time out'}
                print('socket time out')
                result = {'data':a,'ip_port':''}
                connection.send(json.dumps(result))
            except Exception,e:
                a={'request_id':request_id,'img':'' ,'code':'502','message':''}
                print ('call algorithm module api error')
                print(e)
                result = {'data':a,'ip_port':''}
                connection.send(json.dumps(result))
            finally:
                connection.close()


if __name__ == '__main__':  
    help_msg = 'Usage: python %s <start|stop|restart|status>' % sys.argv[0]  
    if len(sys.argv) != 4:
        print(help_msg)
        print(sys.argv)
        sys.exit(1)  
    port=sys.argv[2]
    gpu = sys.argv[3]
    tag='mask_iden_service_{}'.format(port)

    # r = redis.StrictRedis(host='localhost', port=6379, db=0)
    # IDENTITY_CARD_MOSAIC_ALGO = pickle.loads(r.hget('identity_card_mosaic_algo','socket_port_map'))
    # sock=IDENTITY_CARD_MOSAIC_ALGO[port]['net_check_service']
    #sock = "mask_iden"

    pid_fn = '/tmp/%s.pid'%(tag) #process pid
    log_fn = '/tmp/%s.log'%(tag) #process log absolute path 
    err_fn = '/tmp/%s.err.log'%(tag) #process error log
    cD = ClientDaemon(tag,gpu, pid_fn,stdout=log_fn, stderr=err_fn, verbose=1)
  
    if sys.argv[1] == 'start':  
        cD.start(log_fn)  
    elif sys.argv[1] == 'stop':  
        cD.stop()  
    elif sys.argv[1] == 'restart':  
        cD.restart(log_fn)  
    elif sys.argv[1] == 'status':  
        alive = cD.is_running()
        if alive:  
            print('process [%s] is running ......' % cD.get_pid())
        else:  
            print('daemon process [%s] stopped' %cD.name)
    else:  
        print('invalid argument!')
        print(help_msg)
