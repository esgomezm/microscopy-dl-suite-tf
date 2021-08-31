"""
Created on Fri Oct 9 2020

@author: E. Gomez de Mariscal
GitHub username: esgomezm
"""
import sys
import subprocess
import os
import numpy as np

CTCpath = sys.argv[1]
CTCEvaluator = sys.argv[2]
labels = sys.argv[3]
labels = np.int(labels)
task = CTCEvaluator.split()

if CTCEvaluator.__contains__("SEGMeasure"):    
    SEQ = os.listdir(CTCpath)
    SEQ = [folder.split('_')[0] for folder in SEQ if folder.__contains__('RES')]
    SEQ.sort()
    JACCARD = []
    RESULT = []
    if labels>1:
        JACCARD1 = []
        JACCARD2 = []
    
    for s in SEQ:
        if not s=='01':
            if CTCEvaluator.__contains__("Linux"):
                cmd = (CTCEvaluator, CTCpath, s, '3')
            else:
                # Windows
                cmd = (CTCEvaluator, CTCpath, s)
            seg_measure = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].strip()
            seg = seg_measure.decode('utf8')
            JACCARD.append(np.float32(seg.split(':')[-1]))
            RESULT.append('Sequence {} '.format(s) + seg)
            print(RESULT[-1])
            if labels > 1:
                L1 = []
                L2 = []
                LOG = open(os.path.join(CTCpath, s+'_RES', 'SEG_log.txt'), "r")
                list_of_lines = LOG.readlines()
                for line in list_of_lines:
                    if line.__contains__('GT_label=1'):
                        L1.append(np.float(line.split('J=')[-1]))
                    elif line.__contains__('GT_label=2'):
                        L2.append(np.float(line.split('J=')[-1]))
                JACCARD1.append(np.mean(L1))
                JACCARD2.append(np.mean(L2))
                print('Accuracy measure for label = 1: {}'.format(JACCARD1[-1]))
                print('Accuracy measure for label = 2: {}'.format(JACCARD2[-1]))
    
    RESULT.append('Total average: {0} and standard deviation: {1}'.format(np.mean(JACCARD), np.std(JACCARD)))
    if labels > 1:
        RESULT.append('Total average for L1: {0} and standard deviation: {1}'.format(np.mean(JACCARD1), np.std(JACCARD1)))
        RESULT.append('Total average for L2: {0} and standard deviation: {1}'.format(np.mean(JACCARD2), np.std(JACCARD2)))
    with open(os.path.join(CTCpath, 'SEG_results.txt'), 'w') as filehandle:
        for listitem in RESULT:
            filehandle.write('%s\n' % listitem)
elif CTCEvaluator.__contains__("TRAMeasure"):
    SEQ = os.listdir(CTCpath)
    SEQ = [folder.split('_')[0] for folder in SEQ if folder.__contains__('RES')]
    SEQ.sort()
    AOG = []
    RESULT = []
   
    for s in SEQ:
        if not s=='01':
            if CTCEvaluator.__contains__("Linux"):
                cmd = (CTCEvaluator, CTCpath, s, '3')
            else:
                # Windows
                cmd = (CTCEvaluator, CTCpath, s)
            seg_measure = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].strip()
            seg = seg_measure.decode('utf8')
            AOG.append(np.float32(seg.split(':')[-1]))
            RESULT.append('Sequence {} '.format(s) + seg)
            print(RESULT[-1])
   
    RESULT.append('Total average: {0} and standard deviation: {1}'.format(np.mean(AOG), np.std(AOG)))
    
    with open(os.path.join(CTCpath, 'TRA_results.txt'), 'w') as filehandle:
        for listitem in RESULT:
            filehandle.write('%s\n' % listitem)
