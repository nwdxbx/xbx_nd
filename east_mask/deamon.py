#!/usr/bin/env python  
#encoding: utf-8  
#Author:xxx
#Date:2017.04.01
#tips: 
#usage: start:      python daemon_class.py start  
#       stop :      python daemon_class.py stop  
#       status:     python daemon_class.py status  
#       restart:    python daemon_class.py restart  
#       pid info:   ps -axj | grep daemon_class  

import atexit, os, sys, time, signal  
  
class cdaemon:  

    def __init__(self, save_path, stdin=os.devnull, stdout=os.devnull, stderr=os.devnull, home_dir='.', umask=022, verbose=1):  
        self.stdin = stdin  
        self.stdout = stdout  
        self.stderr = stderr  
        self.pidfile = save_path
        self.home_dir = home_dir  
        self.verbose = verbose
        self.umask = umask  
        self.daemon_alive = True  
  
    def daemonize(self):  
        try:  
            pid = os.fork()  
            if pid > 0:  
                sys.exit(0)  
        except OSError, e:  
            sys.stderr.write('fork #1 failed: %d (%s)\n' % (e.errno, e.strerror))  
            sys.exit(1)  
  
        os.chdir(self.home_dir)  
        os.setsid()  
        os.umask(self.umask)  
  
        try:  
            pid = os.fork()  
            if pid > 0:  
                sys.exit(0)  
        except OSError, e:  
            sys.stderr.write('fork #2 failed: %d (%s)\n' % (e.errno, e.strerror))  
            sys.exit(1)  
  
        sys.stdout.flush()  
        sys.stderr.flush()  
  
        si = file(self.stdin, 'r')  
        so = file(self.stdout, 'a+')  
        if self.stderr:  
            se = file(self.stderr, 'a+', 0)  
        else:  
            se = so  
  
        os.dup2(si.fileno(), sys.stdin.fileno())  
        os.dup2(so.fileno(), sys.stdout.fileno())  
        os.dup2(se.fileno(), sys.stderr.fileno())  
  
        def sig_handler(signum, frame):  
            self.daemon_alive = False  
        signal.signal(signal.SIGTERM, sig_handler)  
        signal.signal(signal.SIGINT, sig_handler)  
  
        if self.verbose >= 1:  
            print 'daemon process started ...'  
  
        atexit.register(self.del_pid)  
        pid = str(os.getpid())  
        file(self.pidfile, 'w+').write('%s\n' % pid)  
  
    def get_pid(self):  
        try:  
            pf = file(self.pidfile, 'r')  
            pid = int(pf.read().strip())  
            pf.close()  
        except IOError:  
            pid = None  
        except SystemExit:  
            pid = None  
        return pid  
  
    def del_pid(self):  
        if os.path.exists(self.pidfile):  
            os.remove(self.pidfile)  
  
    def start(self, *args, **kwargs):  
        if self.verbose >= 1:  
            print 'ready to starting ......'  
        #check for a pid file to see if the daemon already runs  
        pid = self.get_pid()  
        if pid:  
            msg = 'pid file %s already exists, is it already running?\n'  
            sys.stderr.write(msg % self.pidfile)  
            sys.exit(1)  
        #start the daemon  
        self.daemonize()  
        self.run(*args, **kwargs)  
  
    def stop(self):  
        if self.verbose >= 1:  
            print 'stopping ...'  
        pid = self.get_pid()  
        if not pid:  
            msg = 'pid file [%s] does not exist. Not running?\n' % self.pidfile  
            sys.stderr.write(msg)  
            if os.path.exists(self.pidfile):  
                os.remove(self.pidfile)  
            return  
        #try to kill the daemon process  
        try:  
            i = 0  
            while 1:  
                os.kill(pid, signal.SIGTERM)  
                time.sleep(0.1)  
                i = i + 1  
                if i % 10 == 0:  
                    os.kill(pid, signal.SIGHUP)  
        except OSError, err:  
            err = str(err)  
            if err.find('No such process') > 0:  
                if os.path.exists(self.pidfile):  
                    os.remove(self.pidfile)  
            else:  
                print str(err)  
                sys.exit(1)  
            if self.verbose >= 1:  
                print 'Stopped!'  
  
    def restart(self, *args, **kwargs):  
        self.stop()  
        self.start(*args, **kwargs)  
  
    def is_running(self):  
        pid = self.get_pid()  
        #print(pid)  
        return pid and os.path.exists('/proc/%d' % pid)  
  
    def run(self, *args, **kwargs):  
        'NOTE: override the method in subclass'  
        print 'base class run()'  
  
