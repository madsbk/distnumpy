#!/usr/bin/python
import sys
import numpy
import getopt
import pickle

#########Processing Script#############
if __name__ == "__main__":
    FIND_SPEEDUP = 0
    MAX_NP = 16 #-1 #Unlimited number of processors

    options, remainders = getopt.gnu_getopt(sys.argv[1:], '', ['max_np='])

    for opt, arg in options:
        if opt in ('--max_np'):
            MAX_NP = int(arg)

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
        ws = i['WORLDSIZE']
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

    #Print CSV file
    print 'CPUs;Runtime(node);Commtime(node)'

    keys=inputs.keys()
    keys.sort()

    for k in keys:
        runtime = numpy.array([v['total'] for v in inputs[k]], dtype=float)
        runtime = int(numpy.average(runtime) / 1000)# in ms
        waittime = numpy.array([v['wait_time'] for v in inputs[k]], dtype=float)
        waittime = int(numpy.average(waittime) / 1000) # in ms
        print "%6d;%8d;%8d"%(k, runtime, waittime)
