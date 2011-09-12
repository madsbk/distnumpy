#!/usr/bin/python
import sys
import numpy
import getopt
import pickle
import os

#########Processing Script#############
if __name__ == "__main__":
    #-1: no speedup, 0: speedup (NumPy), 1+: speedup (DistNumPy)
    FIND_SPEEDUP = -1
    MAX_NP = -1 #Unlimited number of processors
    FLOPexp = None

    options, remainders = getopt.gnu_getopt(sys.argv[1:], '', ['max_np=', 'speedup=', 'flop='])

    for opt, arg in options:
        if opt in ('--max_np'):
            MAX_NP = int(arg)
        if opt in ('--speedup'):
            FIND_SPEEDUP = int(arg)
        if opt in ('--flop'):
            FLOPexp = arg #FLOP calculation expression e.g. "2*n**3"

    #Read dirs
    input = []
    for f in remainders:
        if os.path.isfile(f):
            input.append(f)
        else:
            for i in os.listdir(f):
                fpath = os.path.join(f, i)
                if os.path.isfile(fpath):
                    input.append(fpath)

    #Read input files
    inputfiles = []
    for f in input:
        pkl_file = open(f, 'rb')
        try:
            dict = pickle.load(pkl_file)
            inputfiles.append(dict)
        except:
            pass
        pkl_file.close()

    #Arrange the input files such that inputs with the same worldsize
    #are under the same key.
    inputs = {}
    for i in inputfiles:
        ws = 0
        if i['dist']:
            try:
                ws = i['THREADS']
            except:
                pass
            ws *= i['WORLDSIZE']
        if ws in inputs:
            inputs[ws].append(i)
        else:
            inputs[ws] = [i]

    #Sanity input check
    for value in inputs.values():
        t = numpy.array([v['dist'] for v in value])
        assert numpy.equal(t, t[0]).all()
        t = numpy.array([v['jobsize'] for v in value])
        assert numpy.equal(t, t[0]).all()
        t = numpy.array([v['BLOCKSIZE'] for v in value])
        assert numpy.equal(t, t[0]).all()

    #Clean the input
    for value in inputs.values():
        totals = numpy.array([v['total'] for v in value])
        avg=numpy.average(totals)
        std=numpy.std(totals)
        min=avg-2*std
        max=avg+2*std
        rm = []
        for i in xrange(len(value)):
            if value[i]['total'] < min or value[i]['total'] > max:
                print "# Warning - the result (worldsize: %d, nthreads:"\
                      "%d, runtime: %d ms) is removed because of the "\
                      "standard deviation rule."%(value[i]['WORLDSIZE'],\
                      value[i]['THREADS'], value[i]['total'] / 1000)
                rm.append(i)
        for i in rm:
            value.pop(i)

    #Check for low sample count
    for v in inputs.values():
        if 0 < len(v) < 3:
            print "# Warning - there is only %d runs with a " \
                  "worldsize of %d and nthreads of %d"%(len(v), \
                  v[0]['WORLDSIZE'], v[0]['THREADS'])

    #Compute total waiting and computation time for each input.
    for value in inputs.values():
        for i in value:
            i['wait_time'] = i['ufunc_comm'] + i['comm_init'] + \
                             i['reduce_nd'] - i['reduce_nd_apply'] + \
                             i['msg2slaves'] + i['final_barrier']
            i['comp_time'] = i['apply_ufunc'] + i['reduce_nd_apply']

    #Compute average values.
    result = {}
    for (k,v) in inputs.items():
        run  = numpy.array([i['total'] for i in v], dtype=float)
        wait = numpy.array([i['wait_time'] for i in v], dtype=float)
        comp = numpy.array([i['comp_time'] for i in v], dtype=float)
        result[k] = [v[0]['jobsize'], numpy.average(run),numpy.average(wait),\
                     numpy.average(comp),v[0]['THREADS']]

        if FLOPexp is not None:#Convert to FLOPS
            flop = float(eval(FLOPexp,{"n":i['jobsize']}))
            flops = flop / (result[k][1] / 1000000.0)
            result[k][1] = flops


    #Print CSV file
    keys=result.keys()
    keys.sort()
    if FIND_SPEEDUP > -1: #Show speedup instead of runtime.
        speedup = result[FIND_SPEEDUP][1]#The speedup baseline
        print '  CPUs;Threads; Jobsize; Speedup;Commtime;Comptime'
        for k in keys:
            if FLOPexp is None:
                runtime  = speedup / result[k][1]
            else:
                runtime  = result[k][1] / speedup
            waittime = int(result[k][2] / 1000)# convert to ms
            comptime = int(result[k][3] / 1000)# convert to ms
            print "%6d;%7d;%8d;%8.3f;%8d;%8d"%(k, result[k][4], result[k][0], runtime, waittime, comptime)
    else:
        if FLOPexp is None:
            print '  CPUs;Threads; Jobsize; Runtime;Commtime;Comptime'
        else:
            print '  CPUs;Threads; Jobsize;  GFLOPS;Commtime;Comptime'

        if 0 in result:#Sequential runtime
            if FLOPexp is None:
                print "#  Seq;%7d;%8d;%8d;"%(result[0][4],result[0][0],result[0][1]/1000)
            else:
                print "#  Seq;%7d;%8d;%8.3f;"%(result[0][4],result[0][0],result[0][1]/2**30)
            keys.pop(0)#Remote the zero key.

        for k in keys:
            waittime = int(result[k][2] / 1000)# convert to ms
            comptime = int(result[k][3] / 1000)# convert to ms
            if FLOPexp is None:
                runtime  = int(result[k][1] / 1000)# convert to ms
                print "%6d;%7d;%8d;%8d;%8d;%8d"%(k, result[k][4], result[k][0], runtime, waittime, comptime)
            else:
                runtime  = result[k][1] / 2**30# convert to GFLOPS
                print "%6d;%7d;%8d;%8.3f;%8d;%8d"%(k, result[k][4], result[k][0], runtime, waittime, comptime)


