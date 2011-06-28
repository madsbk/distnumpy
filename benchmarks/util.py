#Benchmarks for DistNumPy.
#This is collection of help functions for the DistNumPy benchmarks.

import numpy as np
import getopt

class Parsing:
    """This class should handle the presentation of benchmark results.
       It is initiated with the benchmark arguments 'args'.
       A list of non-optional arguments is exposed through self.argv.
    """
    def __init__(self, args):
        self.runinfo = {}
        self.dist = False
        self.filename = "benchmark_dump.pkl"
        options, self.argv = getopt.gnu_getopt(args, 'd:n:c:f:', ['dist=','nnodes=','ncores=','filename='])

        for opt, arg in options:
            if opt in ('-d', '--dist'):
                self.dist = bool(eval(arg))
            if opt in ('-n', '--nnodes'):
                self.runinfo['nnodes'] = int(arg)
            if opt in ('-c', '--ncores'):
                self.runinfo['ncores'] = int(arg)
            if opt in ('-f', '--filename'):
                self.filename = arg

    def pprint(self, timing, sysinfo=True, rank=0):
        """Pretty-print the timing profile.
           System info, such as worldsize and process rank, can be included.
           If timing is a tuple the dictionary in position 'rank' is used.
        """
        if isinstance(timing, tuple):
            timing = timing[rank] #Default: printing rank zero.

        print "*******RUNTIME INFO*******"
        print "dist: %d"%(self.dist)
        l = self.runinfo.items();l.reverse()
        for k, v in l:
            print "%s: %d"%(k, v)

        if sysinfo:
            print "*********SYS INFO*********"
            print "SPMD_MODE: %d"%np.SPMD_MODE
            print "RANK:      %d"%np.RANK
            print "BLOCKSIZE: %d"%np.BLOCKSIZE

        print "**********TIMING**********"
        print "Total:           %7.d ms"%(timing['total'] / 1000)
        print "  dag_svb_flush: %7.d ms"%(timing['dag_svb_flush'] / 1000)
        print "    dag_svb_rm:  %7.d ms"%(timing['dag_svb_rm'] / 1000)
        print "    apply_ufunc: %7.d ms"%(timing['apply_ufunc'] / 1000)
        print "    ufunc_comm:  %7.d ms"%(timing['ufunc_comm'] / 1000)
        print "    comm_init:   %7.d ms"%(timing['comm_init'] / 1000)
        print "    data_free:   %7.d ms"%(timing['arydata_free'] / 1000)
        print "  reduce (1-d):  %7.d ms"%(timing['reduce_1d'] / 1000)
        print "  reduce (n-d):  %7.d ms"%(timing['reduce_nd'] / 1000)
        print "    pre_apply:   %7.d ms"%(timing['reduce_nd_apply'] / 1000)
        print "  zerofill:      %7.d ms"%(timing['zerofill'] / 1000)
        print "  ufunc_svb:     %7.d ms"%(timing['ufunc_svb'] / 1000)
        print "    dag_svb_add: %7.d ms"%(timing['dag_svb_add'] / 1000)
        print "    calc_vblock: %7.d ms"%(timing['calc_vblock'] / 1000)
        print "  data_malloc:   %7.d ms"%(timing['arydata_malloc'] / 1000)
        print "  msg2slaves:    %7.d ms"%(timing['msg2slaves'] / 1000)
        print "*********COUNTERS*********"
        print "nflush:         %8.d"%(timing['nflush'])
        print "mem_reused:     %8.d"%(timing['mem_reused'])
        print "nconnect:       %8.d"%(timing['nconnect'])
        print "nconnect_max:   %8.d"%(timing['nconnect_max'])
        if timing['napply'] == 0:#Avoid ZeroDivisionError
            nconnect_avg = 0.0;
        else:
            nconnect_avg = timing['nconnect'] / float(timing['napply'])
        print "nconnect_avg:      %8.2f"%(nconnect_avg)
        print "**************************"

    def write_dict(self, timing, filename=None, sysinfo=True, rank=0):
        """Writes the timing profile as a pickled Python dictionary.
           If filename is None the application option
           System info, such as worldsize and process rank, can be included.
           If timing is a tuple the dictionary in position 'rank' is used.
        """
        if isinstance(timing, tuple):
            timing = timing[rank] #Default: printing rank zero.
        if filename is None:
            filename = self.filename

        ret = dict(timing.items() + self.runinfo.items())

        if sysinfo:
            ret['SPMD_MODE'] = np.SPMD_MODE
            ret['RANK'] = np.RANK
            ret['BLOCKSIZE'] = np.BLOCKSIZE

        import pickle
        output = open(filename, 'wb')
        pickle.dump(ret, output)
        output.close()

