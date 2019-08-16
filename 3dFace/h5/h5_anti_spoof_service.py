import tornado.web
import tornado.ioloop
from utils.deamon import cdaemon
from utils import h5_logger
from utils import cfg
import h5AntiSpoof_interface

import json
from timeit import default_timer as timer


class ClientDaemon(cdaemon):
    def __init__(self,mylog, service_name, gpu, port, save_path, stdin=os.devnull, stdout=os.devnull, stderr=os.devnull, home_dir='.', umask=022, verbose=1):
        self.service_name = service_name
        self.gpu = int(gpu)
        self.port = port
        self.log = mylog
        cdaemon.__init__(self, save_path, stdin, stdout, stderr, home_dir, umask, verbose)

    class AlgoService(tornado.web.RequestHandler):
        def post(self):
            start = timer()
            data = tornado.escape.json_decode(self.request.body)
            end = timer()
            ClientDaemon.algo.log.writelog("REQUEST DATA",json.dumps({"flag":"*****************"}))
            ClientDaemon.algo.log.writelog("REQUEST TIME",json.dumps({"time":str(end-start)}))
            start = timer()
            result = ClientDaemon.algo.get_verification_result(data)
            ClientDaemon.algo.log.writelog("ALGORITHM TIME",json.dumps({"time":str(end-start)}))
            self.write(result)
            ClientDaemon.algo.log.writelog("POST DATA",json.dumps({"flag":"*****************"}))
    def run(self,output_fn,**kwargs):
        ClientDaemon.algo = h5AntiSpoof_interface.H5AntiSpoof(self.log,self.gpu,self.port)
        ClientDaemon.algo.log.writelog("INIT FINISH",json.dumps({"flag":"model init finish."}))
        app = tornado.web.Application([("/200/",self.AlgoService),])
        app.listen(self.port)
        tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    help_msg = "Usage: python %s <start|stop|restart|status>" % sys.argv[0]
    if len(sys.argv)!=4:
        print(help_msg)
        sys.exit(1)
    port = sys.argv[2]
    gpu = sys.argv[3]
    service_name = "{}_service_{}".format(cfg.MODULE_NAME,port)

    pid_fn = "/tmp/%s.pid" % service_name
    log_fn = "/tmp/%s.log" % service_name
    err_fn = "/tmp/%s.err.log" % service_name

    log = h5_logger.Logger(port)
    cD = ClientDaemon(log,service_name, gpu, port, pid_fn, stdout=log_fn, stderr=err_fn)

    if sys.argv[1] == "start":
        cD.start(log_fn)
    elif sys.argv[1] == "stop":
        cD.stop()
    elif sys.argv[1] == "restart":
        cD.restart(log_fn)
    elif sys.argv[1] == "status":
        alive = cD.is_running()
        if alive:
            print("process [%s] is running ......" % cD.get_pid())
        else:
            print("daemon process [%s] stopped" %cD.name)
    else:
        print("invalid argument!")
        print(help_msg)