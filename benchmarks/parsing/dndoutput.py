#!/usr/bin/python
import sys
import numpy
import os

class stats(): #Container for statistics
    def __init__(self, avg, min, max, var, std):
        self.avg=avg
        self.min=min
        self.max=max
        self.var=var
        self.std=std

class result(): #Container final results
    def __init__(self, run, wait):
        self.run=run
        self.wait=wait


def parse(fname, results): #Parse input file all words that end with : are
    file=open(fname,'r')   #considered keys in a dictionary with the next entry
    data=file.readlines()  #as the value
    file.close()

    dict = {}
    target = None

    for line in data:
        elem = line.split()
        for i in range(len(elem)):
            if elem[i]=='(Non-Dist)':target='seq'
            else:
                if elem[i][-1]==':':
                    key=elem[i][:-1]
                    try:
                        dict[key]=elem[i+1]
                    except:
                        pass #maybe there is no data :)

    if not target:
        try:
            target=dict['notes']
        except:
            print '# Unable to determine run_name of',fname,'dropping file'
            return

    if results.has_key(target):
        for e in dict.keys():
            if results[target].has_key(e):
                try:
                    results[target][e].append(float(dict[e]))
                except: pass
            else:
                try:
                    results[target][e]=[float(dict[e])]
                except: pass

    else:
        results[target]={}
        for e in dict.keys():
            try:
                results[target][e]=[float(dict[e])]
            except:
                results[target][e]=[]

    return results

def get_bad(data):
    num=numpy.array(data)
    avg=numpy.average(num)
    std=numpy.std(num)
    min=avg-2*std
    max=avg+2*std

    res = []

    for i in range(len(data)):
        if not min<=data[i]<=max:
            res.append(i)

    return res

def clean_data(target, key):
    bad=get_bad(target[key])
    for k in target.keys():
        for b in bad:
            try:
                target[k].pop(b)
            except:
                pass

def calc_data(dict):
    result={}
    for key in dict.keys():
        try:
            data=numpy.array(dict[key])
            avg=numpy.average(data)
            min=numpy.min(data)
            max=numpy.max(data)
            var=numpy.var(data)
            std=numpy.std(data)
            result[key]=stats(avg, min, max, var, std)
        except Exception, e:
            result[key]=stats(.0, .0, .0, .0, .0)

    return result

#########Processing Script#############
if __name__ == "__main__":
    MERGED_NAME = True
    FIND_SPEEDUP = 0
    MAX_NP = 16 #-1 #Unlimited number of processors
    assert len(sys.argv) == 3, "the input directory is not specifed"
    if sys.argv[1] == "numpy":
        FIND_SPEEDUP = 1
    elif sys.argv[1] == "dnumpy":
        FIND_SPEEDUP = 2


    inputs = [os.path.join(sys.argv[2], i) for i in os.listdir(sys.argv[2]) if os.path.isfile(os.path.join(sys.argv[2], i))]
    results = {}
    for i in inputs:
        parse(i, results)

    core={}
    node={}

    #Now the data has been cleaned
    max_nodes=1
    max_cores=1

    #Clean up data - eliminate bad messurements
    for key in results.keys():
        clean_data(results[key],'app_total') #We eliminate experiments with too
                                              #large variase on execution time

    #Now the data has been cleaned
    for key in results.keys():
        if key=='seq':
            results[key]['nodes']=1
            results[key]['cores']=1
            results[key]['procs'] = 'seq'
        else:
            if MERGED_NAME:                       #Old naming - without cores and nodes
                data = key.split('_')             #as keys
                nodes=int(data[1])
                cores=int(data[2])
            else:
                nodes=mes['nodes'][0]
                cores=mes['cores'][0]

            results[key]['nodes']=nodes
            results[key]['cores']=cores
            results[key]['procs']=nodes*cores

            max_nodes=max(nodes, max_nodes)
            max_cores=max(cores, max_cores)

    for key in results.keys():
        cur=calc_data(results[key])

        nodes=results[key]['nodes']
        cores=results[key]['cores']
        procs=results[key]['procs']

        wait_time=cur['ufunc_comm'].avg + cur['comm_init'].avg +cur['(n-d)'].avg-cur['pre_apply'].avg+cur['msg2slaves'].avg
        if nodes>cores or nodes==max_nodes or procs=='seq':
            node[procs]=result(cur['app_total'].avg, wait_time)
        if cores>nodes or cores==max_cores or procs=='seq':
            core[procs]=result(cur['app_total'].avg, wait_time)


    keys=node.keys()
    keys.sort()
    print '#CPUs;Runtime(node);Commtime(node);Runtime(core);Overhead(core)'

    bycore = True
    if FIND_SPEEDUP:
        if FIND_SPEEDUP == 2:
            seq=node[2].run*2.
        else:
            seq=node['seq'].run
        for key in node.keys():
            node[key].wait=node[key].wait*100/node[key].run
            node[key].run=seq/node[key].run
            try:
                core[key].wait=core[key].wait*100/core[key].run
                core[key].run=seq/core[key].run
            except:
                bycore = False

    key='seq'
    print '#',key,';',node[key].run,';',node[key].wait,';',
    if bycore:
        print core[key].run,';',core[key].wait,';',
    print ""
    for key in keys[:-1]:
        if MAX_NP == -1 or key <= MAX_NP:
            print key,';',node[key].run,';',node[key].wait,';',
            if bycore:
                print core[key].run,';',core[key].wait,';',
            print ""
