#Bayesian Updating
#Ascii-Marker Parsing

#Output File of form
#1: r 0 1 1 1 
#meaning decision for the rare urn after the balls 0 1 1 1 (0 = frequent and 1 = rare balls)

import os
import re

#epoch criterion: stimulus S 4 is the first stimulus 10-2000ms after S 1/S 2

def filter_epoch(filename):
    """does the same as filter_file but checks for epoch criterion"""
    #code only 1/2 instead of S and -1/-2 if S doesnt fulfill epoch criteria, tableau for R 1
    #sampling interval = 4ms
    smi = 4
    last = (0,0)
    
    with open(filename) as fl:       
        for line in fl:
             splitted = line.split(",")
             if len(splitted) > 1:
                 if " S " in splitted[1]:
                     #check if it fulfills epoch criteria
                     if " 4" in splitted[1]:
                         if last[1] != 0:
                             diff = int(splitted[2])-last[1]
                             if diff > 3 and diff <= 2000/smi:
                                 yield last[0]
                                 last = (0,0)
                             else:
                                 yield -last[0]
                                 last = (0,0)
                     
                     if " 2" in splitted[1] or " 1" in splitted[1]:
                         last = (int(splitted[1][-1]),int(splitted[2]))
                 if " R " in splitted[1]:
                     #return "tableau" if R 1
                     if last[1] != 0:
                         yield -last[0]
                         last = (0,0)
                     if " 1" in splitted[1]:
                         yield "tableau"
                         
                     
             
                     
                     
         
                 


def parse_for_epoch(filename,stim_dict):
    """
    filters all lines of filename, returns a vector, 
    containing all S 1/S 2 coded as 1 freq and 2 rare balls who fulfill the epoch criteria
    and -1/-2 if they dont fulfill it
    """
    #probably a good idea to exclude all sequences with a missing epoch
    #but first dont do that
    #TODO: how to present failed criterion epochs, how to write them
    stimuli = []
    
    for event in filter_epoch(filename):
        if event == "tableau":
            if stimuli != []:
                stimuli = []
        else:
            stimuli.append(stim_dict[event])
            yield " ".join(map(str,stimuli))
        
   
         
            
        
        

def filter_vmkr(filename):
    """
    filters out all lines except S 1/S 2 in a vmkr file
    """
    #TODO: this is ugly, change it
    with open(filename) as fl:
        return [ line.split(",")[1] for line in fl if len(line.split(","))>1 and ((line.split(",")[1]=="S  2") or ((line.split(",")[1]=="S  1"))) ]

def filter_file(filename,epoch_check=False):
    '''
    filters out all unnecessary lines, explicitly lines which dont contain S 1, S 2, R 1, R 2, R 4, as strings
    if epoch_check = True,      
    '''
    #iterate through every line, discard lines without meaning
    #discard lines which are not stimulus1/2 or response1/2/4 (write response 1 as 'tableau'
    
    file = open(filename)
    
    real_lines = list()
    
    for line in file:
        splitted = line.split()
        if splitted[1] == "S":
            if splitted[2] == "1," or splitted[2] == "2,":
                real_lines.append(splitted[1] + " " + splitted[2][0])
            else:
                continue
        elif splitted[1] == "R":
            if splitted[2] == "2," or splitted[2] == "4,":
                real_lines.append(splitted[1] + " " + splitted[2][0])
            elif splitted[2] == "1,":
                real_lines.append("tableau")
            else:
                continue
        else:
            continue
        
    return real_lines
    
def parse_file(filename,stim_dict,resp_dict):
    '''
    parses whole file after opening file at filename
    parses the line line with S1 = rare ball stimulus/S2 = freq ball stimulus/R2 = freq urn dec/R4 = rare urn dec
    then returns a string of the form
    decision [r/f] - last ball [0/1]
    '''
    filter_lines = filter_file(filename)
    
    #now all unncessary lines are filtered out
    #create cycles with choices
    #just for now only creates strings consisting of decision r/f and the balls
    #change this later for reihenfolge-effekt untersuchung
    
    #decision_lines = list()
    stimuli = []
    
    for line in filter_lines:
        if line == 'tableau':
            if stimuli != []:
                stimuli = []
        else:
            splitted = line.split()
            if splitted[0] == 'S':
                 stimuli.append(stim_dict[splitted[1]])
            elif splitted[0] == 'R':
                yield (resp_dict[splitted[1]] + "".join(map(lambda a,b : a+b,[' ']*len(stimuli),stimuli)))
    

def aggregate_decisions(data):
    '''
    aggregates to a data format fit for processing in a binomial distribution.
    returns list with 14 entries, each representing the decisions for the rare urn for the number of rare balls + i = range(5) as its index
    '''
    list_of_values = [0]*14
    list_of_values2 = [0]*14
    for line in data:
        splitted = (line.strip()).split()
        list_of_values[sum(int(i) for i in splitted[1:])+([0,2,5,9][len(splitted)-2])] += 1 if splitted[0] == 'r' else 0 
        list_of_values2[sum(int(i) for i in splitted[1:])+([0,2,5,9][len(splitted)-2])] += 1
    
    return [ "{0} {1}\n".format(i,k) for i,k in zip(list_of_values,list_of_values2) ]


#%%
path = "/home/mboos/Work/Bayesian Updating/Data/"
#be aware of the impliciations: following the counter-balancing by flo&caro

epoch_stim_dicts = [{1 : 2, 2 : 1,-1:-2,-2:-1}]*8 + [{1 : 1 , 2 : 2,-1:-1,-2:-2}]*8

stim_dicts = [{"1" : "1", "2" : "0"}]*8 + [{"1" : "0" , "2" : "1"}]*8
resp_dicts = [{"2" : "f", "4" : "r"}]*4 + [{"2" : "r", "4" : "f"}]*4 + [{"2" : "f", "4" : "r"}]*4 + [{"2" : "r", "4" : "f"}]*4
#%%
#outdated
#==============================================================================
# for filename in os.listdir(path):
#     if filename == 'newdata':
#         continue
#     with open(path+"newdata/"+"ND_"+filename,'w') as new_f:
#         for line in parse_file(path+filename):
#             new_f.write(line+"\n")
#==============================================================================


#%%
for filename in os.listdir(path):
    if filename == 'newdata':
        continue
    else:
        m = re.search('(?<=VP)[0-1][0-9]',filename)
        with open(path+"newdata/"+"ND_"+filename,'w') as new_f:
            new_f.writelines(aggregate_decisions([line for line in parse_file(path+filename,stim_dicts[int(m.group(0))-1],resp_dicts[int(m.group(0))-1])]))


#%%
#redundant?
#TODO: parse specifically for each balls
#test for each ball
#Need length data: 
filename = "VP01_50exp_Raw Data_markers.txt"
with open(path+"EB_"+filename,"w") as new_f:
    new_f.writelines([line+"\n" for line in parse_file(path+filename,stim_dicts[0],resp_dicts[0])])
    
#%%

#path = "/home/mboos/Work/Bayesian Updating/Data EEG/"
#filename = "VP01_50exp_OverallSegmentation.vmrk"
#with open(path+"EB_"+filename,"w") as new_f:

#TODO: TS for all Marker files

path = "/home/mboos/Work/Bayesian Updating/Data/"
#filename = "VP01_50exp_Raw Data_markers.txt"
#counter-balancing flo/caro
epoch_stim_dicts = [{1 : 2, 2 : 1,-1:-2,-2:-1}]*8 + [{1 : 1 , 2 : 2,-1:-1,-2:-2}]*8

files = os.listdir(path)

#maybe even without pattern
pattern = "80exp"

#test this!
for fn in files:
    if fn.startswith("VP") and pattern in fn:
        with open(path+"TS"+fn,"w") as new_f:
            for ln in parse_for_epoch(path+fn,epoch_stim_dicts[int(fn[2:4])-1]):
                new_f.write(ln+"\n")
            
#%%
with open(path+"TStestTS.txt","w") as new_f:
    for ln in parse_for_epoch(path+"testTS.txt",epoch_stim_dicts[13]):
        new_f.write(ln+"\n")
