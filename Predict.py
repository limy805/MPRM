import pickle
import random
from Preprocess import Getdata
import json
import numpy as np
from pathlib import Path
from collections import deque
def getnewlist(L:list):
    Radd = [(t[0],t[1]+1,t[2]) for t in L]
    return Radd
def sample_entity(data_list,num):
    return random.sample(data_list, num)
def Predict(L,rules,hr_t,Rulepath,mode = "Sum"):
    right = 0
    h10, h3, h1, MRR = 0, 0, 0, 0
    Tolnum = len(L)
    allrank = 0
    rankdict = {}
    for numtest, fact in enumerate(L):
        result = {}
        currentrule = rules[fact[1]]
        for pindex,(rule, conf) in enumerate(currentrule):
            lasthopentity = {}
            queue = deque()
            queue.append([fact[0]])
            #rule-based reasoning
            for i, action in enumerate(rule):
                level_size = len(queue)
                for _ in range(level_size):
                    current_path = queue.popleft()
                    current_node = current_path[-1]
                    if (current_node, action) not in hr_t:
                        continue
                    childs = hr_t[(current_node, action)]
                    if len(childs) > bet:
                        childs = sample_entity(childs,bet)
                    for c in childs:
                        #acycle
                        if c in current_path:
                            continue
                        new_path = current_path + [c]
                        queue.append(new_path)
            #record last hop entity s_T
            for entityroad in queue:
                coe = 1
                for index, rela in enumerate(rule):
                    coe = coe * len(hr_t[(entityroad[index], rule[index])])
                if entityroad[-1] not in lasthopentity:
                    lasthopentity[entityroad[-1]] = 0
                lasthopentity[entityroad[-1]] += 1/coe
            #Calculate score
            for lastent,prob in lasthopentity.items():
                if lastent not in result:
                    result[lastent] = 0
                # Max aggregation
                if mode!="Sum" and prob != 0 and len(lasthopentity)>1:
                    tempscore = prob * conf
                    # tempscore = 1 * conf
                    if  tempscore > result[lastent]:
                        result[lastent] = tempscore
                # Sum aggregation
                else:
                    result[lastent] += prob * conf
        #Calculate MRR and Hit10
        if fact[2] not in result:
            continue
        sorted_dict = sorted(result.items(), key=lambda x: x[1], reverse=True)
        #filter
        if (fact[0], fact[1]) in hr_t.keys():
            resultlist = [key for key, _ in sorted_dict if key not in hr_t[(fact[0], fact[1])] ]
            pass
        else:
            resultlist = [key for key, _ in sorted_dict]
        if fact[2] in resultlist:
            right += 1
            rank = next((index + 1 for index, key in enumerate(resultlist) if key == fact[2]), None)
            rankdict[fact] = rank
            allrank += rank
            if rank <= 10:
                h10 += 1
                if rank <= 3:
                    h3 += 1
                    if rank == 1:
                        h1 += 1
            MRR = MRR + 1 / rank
        if numtest % 100 == 0:
            print("Test completion rate:", numtest / Tolnum)
    h10 = round(h10 / Tolnum, 3)
    h3 = round(h3 / Tolnum, 3)
    h1 = round(h1 / Tolnum, 3)
    MRR = round(MRR / Tolnum, 3)
    print(right / len(L),
          "Predictable quantity:{0}\nh10:{1}\nh3:{2}\nh1:{3}\nMRR:{4}".format(right, h10,h3 ,h1 , MRR ))
    with open(Rulepath,"w") as f:
        json.dump({"h10":h10,"h3":h3,"h1":h1,"MRR":MRR},f)
    print("Mean rank:",allrank/right)
def Generate_hr_t(trainlist,tol_r):
    hr_t = {}
    number_r = np.zeros(tol_r)
    for index,t in enumerate(trainlist):
        number_r[t[1]-1] += 1
        if (t[0],t[1]) not in hr_t.keys():
            hr_t[(t[0],t[1])] = set()
        if (t[2],-t[1]) not in hr_t.keys():
            hr_t[(t[2],-t[1])] = set()
        hr_t[(t[2], -t[1])].add(t[0])
        hr_t[(t[0], t[1])].add(t[2])
    return hr_t
def Get_rule(dataname,alpha ,beta ,maxrulelen,maxRulenum = 100):
    dataname = dataname
    Rulename = str(maxrulelen) + "_" + str(alpha) + "_" + str(beta) + "result"
    pathname = Path("Rule") / dataname / Rulename
    with open(pathname, "rb") as f:
        Rule = pickle.load(f)
    for k, v in Rule.items():
        Rule[k] = Rule[k][:maxRulenum]
    return Rule
if __name__ == "__main__":
    random.seed(10)
    Dataname = "YAGO3-10"  #6,300,100,300
    # Dataname = "NELL-995"  # 3,300,100,300
    # Dataname = "YAGO3-10"  # 3,300,100,300
    # Dataname = "FB15K-237"  # 3,300,200,300
    trainlist, testlist, entity_dict, relation_dict = Getdata(Dataname)
    r_tol = len(relation_dict)
    T,alp,bet,Rulenum, = 3,300,100,300
    Rule = Get_rule(dataname = Dataname,alpha = alp, beta = bet ,maxrulelen = T,maxRulenum=Rulenum)
    trainlist = getnewlist(trainlist)
    testlist = getnewlist(testlist)
    hr_t = Generate_hr_t(trainlist,r_tol)
    ResultName = Dataname + "_" + str(T) + "_" + str(alp) + "_" + str(bet) + "_" + str(Rulenum) + "MRR"
    PATH = Path("MRR") /  ResultName
    Predict(testlist, Rule, hr_t,PATH)
