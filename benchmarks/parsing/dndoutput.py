#!/usr/bin/python
import sys
import numpy
import getopt
import pickle

#########Processing Script#############
if __name__ == "__main__":
    #-1: no speedup, 0: speedup (NumPy), 1+: speedup (DistNumPy)
    FIND_SPEEDUP = -1
    MAX_NP = 16 #-1 #Unlimited number of processors

    options, remainders = getopt.gnu_getopt(sys.argv[1:], '', ['max_np=', 'speedup='])

    for opt, arg in options:
        if opt in ('--max_np'):
            MAX_NP = int(arg)
        if opt in ('--speedup'):
            FIND_SPEEDUP = int(arg)

    #Read input files
    inputfiles = []
    for f in remainders:
        pkl_file = open(f, 'rb')
        dict = pickle.load(pkl_file)
        inputfiles.append(dict)
        pkl_file.close()

    #Arrange the input files such that inputs with the same worldsize
    #are under the same key.
    inputs = {}
    for i in inputfiles:
        if i['dist']:
            ws = i['WORLDSIZE']
        else:
            ws = 0 #Zero means sequential execution.
        if ws in inputs:
            inputs[ws].append(i)
        else:
            inputs[ws] = [i]

    #Clean the input
    for value in inputs.values():
        totals = numpy.array([v['total'] for v in value])
        avg=numpy.average(totals)
        std=numpy.std(totals)
        min=avg-2*std
        max=avg+2*std
        for i in xrange(len(value)):
            if value[i]['total'] < min or value[i]['total'] > max:
                print "# Warning - the result (worldsize: %d, runtime:"\
                      " %d ms) is removed because of the standard "\
                      "deviation rule."%(value[i]['WORLDSIZE'], \
                                         value[i]['total'] / 1000)
                value.pop(i)

    #Check for low sample count
    for v in inputs.values():
        if 0 < len(v) < 3:
            print "# Warning - there is only %d runs with a " \
                  "worldsize of %d"%(len(v), v[0]['WORLDSIZE'])

    #Compute total wait time for each input.
    for value in inputs.values():
        for i in value:
            i['wait_time'] = i['ufunc_comm'] + i['comm_init'] + \
                             i['reduce_nd'] - i['reduce_nd_apply'] + \
                             i['msg2slaves']

    #Compute average values.
    result = {}
    for (k,v) in inputs.items():
        run  = numpy.array([i['total'] for i in v], dtype=float)
        wait = numpy.array([i['wait_time'] for i in v], dtype=float)
        result[k] = (numpy.average(run),numpy.average(wait))

    #Print CSV file
    keys=result.keys()
    keys.sort()
    if 0 in result:#Sequential runtime
        SEQ = "#  Seq;%8d;\n"%(result[0][0]/1000)
        keys.pop(0)#Remote the zero key.
    else:
        SEQ = ""
    if FIND_SPEEDUP > -1: #Show speedup instead of runtime.
        speedup = result[FIND_SPEEDUP][0]#The speedup baseline
        print 'CPUs;Speedup;Commtime'
        print SEQ,
        for k in keys:
            runtime  = speedup / result[k][0]
            waittime = int(result[k][1] / 1000)# convert to in ms
            print "%6d;%8.3f;%8d"%(k, runtime, waittime)
    else:
        print 'CPUs;Runtime;Commtime'
        print SEQ,
        for k in keys:
            runtime  = int(result[k][0] / 1000)# convert to in ms
            waittime = int(result[k][1] / 1000)# convert to in ms
            print "%6d;%8d;%8d"%(k, runtime, waittime)

