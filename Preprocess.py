import json
import pickle
from pathlib import Path
def Trainprocess(path):
    with open(path, "r",errors="replace") as file:
        entity_dict = {}
        relation_dict = {}
        index,r_index = 0,0
        triplelist=[]
        for line in file:
            separated_line = line.strip().split()
            if separated_line[0] not in entity_dict:
                entity_dict[separated_line[0]] = index
                index = index + 1
            if separated_line[2] not in entity_dict:
                entity_dict[separated_line[2]] = index
                index = index + 1
            if separated_line[1] not in relation_dict:
                relation_dict[separated_line[1]] = r_index
                r_index = r_index + 1
            triplelist.append((entity_dict[separated_line[0]],relation_dict[separated_line[1]],entity_dict[separated_line[2]]))
        return entity_dict,relation_dict,triplelist
def Testprocess(path,entity_dict,relation_dict):
    triplelist = []
    with open(path, "r",errors="replace") as file:
        for line in file:
            separated_line = line.strip().split()
            if separated_line[0] in entity_dict and separated_line[2] in entity_dict:
                triplelist.append(
                    (entity_dict[separated_line[0]], relation_dict[separated_line[1]], entity_dict[separated_line[2]]))
        return triplelist
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