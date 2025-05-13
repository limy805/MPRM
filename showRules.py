import pickle
import json
from pathlib import Path
def Getdata(Dataname):
    trainpath = Path("data") / Dataname / "train.pkl"
    testpath = Path("data") / Dataname / "test.pkl"
    e_path = Path("data") / Dataname / "e_dict.json"
    r_path = Path("data") / Dataname / "r_dict.json"
    with open(trainpath,"rb") as f:
        trainlist = pickle.load(f)
    with open(testpath, "rb") as f:
        testlist = pickle.load(f)
    with open(e_path,"r") as f:
        entity_dict = json.load(f)
    with open(r_path, "r") as f:
        relation_dict = json.load(f)
    return trainlist,testlist,entity_dict,relation_dict
def Get_rule(dataname,alpha ,beta ,maxrulelen,maxRulenum = 100):
    dataname = dataname
    Rulename = str(maxrulelen) + "_" + str(alpha) + "_" + str(beta) + "result"
    memoryname = str(maxrulelen) + "_" + str(alpha) + "_" + str(beta) + "memory"
    pathname = Path("Rule") / dataname / Rulename
    with open(pathname, "rb") as f:
        Rule = pickle.load(f)
    for k, v in Rule.items():
        Rule[k] = Rule[k][:maxRulenum]
    return Rule
if __name__ == "__main__":
    Dataname = "WN18RR"
    trainlist, testlist, entity_dict, relation_dict = Getdata(Dataname)
    relation_dict = {v+1:k for k,v in relation_dict.items()}
    r_tol = len(relation_dict)
    T, alp, bet, Rulenum, = 6, 300, 100, 5
    Rule = Get_rule(dataname=Dataname, alpha=alp, beta=bet, maxrulelen=T, maxRulenum=Rulenum)
    for Relation,rules in Rule.items():
        print("----------------",Relation-1,"--------------")
        for rule in rules:
            for index,r in enumerate(rule[0]):
                inverse = 0
                if r>0:
                    print(relation_dict[r],end="")
                else:
                    print(relation_dict[-r],end="")
                    inverse=1
                if index==0:
                    if inverse == 1 and len(rule[0]) > 1:
                        print("(z1,x)",end="")
                    elif inverse == 1 and len(rule[0]) == 1:
                        print("(y,x)", end="")
                    elif inverse == 0 and  len(rule[0]) > 1:
                        print("(x,z1)", end="")
                    elif inverse == 0 and len(rule[0]) == 1:
                        print("(x,y)", end="")
                elif index==len(rule[0])-1:
                    if inverse == 1:
                        print("(y,z{0})".format(index),end="")
                    elif inverse == 0:
                        print("(z{0},y)".format(index), end="")
                elif index!=len(rule[0])-1:
                    if inverse == 1:
                        print("(z{0},z{1})".format(index+1,index),end="")
                    elif inverse == 0:
                        print("(z{0},z{1})".format(index,index+1), end="")

                if index!=len(rule[0])-1:
                    print("∧",end="")
            print("⟹",relation_dict[Relation],end="(x,y),")
            print(" " ,rule[1])
        print("\n")