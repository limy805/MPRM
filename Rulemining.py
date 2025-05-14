from Preprocess import Getdata
import random
import heapq
import pickle
from collections import deque, defaultdict
from pathlib import Path
import math
import time
from contextlib import contextmanager
import copy

class MPRM():
    def __init__(self,trainlist,enum,rnum,T,maxfact,maxnei=100):
        self.new_train = trainlist
        self.enum = enum
        self.rnum = rnum
        self.update()
        #hr_t: key:(subject,relation) value:objects
        self.hr_t = {}
        self.rule = {}
        self.samplenum = maxfact
        self.pathlenth = T
        self.graph = {e: set() for e in range(self.enum)}
        # ht_r: key:(subject,object) value:relations
        self.segtrain,self.hr_t,self.ht_r = self.preproc()
        self.Rules = {r + 1:  {}  for r in range(self.rnum)}
        #current fact
        self.current = ()
        self.max_answer = 0
        self.maxnei = maxnei
        print("prepare success")
    @staticmethod
    def get_top_items(dictionary, top_n):
        return heapq.nlargest(top_n, dictionary.items(), key=lambda x: x[1])
    def Chos(self,L,entity):
        choose_r = random.choice(L)
        min_proba = len(self.hr_t[(entity,choose_r)])
        for r in L:
            if len(self.hr_t[(entity,r)]) < min_proba:
                choose_r = r
                min_proba = len(self.hr_t[(entity,r)])
        return choose_r
    @staticmethod
    def has_common_elements(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        common = set1.intersection(set2)
        return len(common) > 0
    def sample_facts(self,data_list):
        if not data_list:
            return []
        sample_size = min(self.samplenum, len(data_list))
        return random.sample(data_list, sample_size)
    def update(self):
        newtrain = []
        for t in self.new_train:
            newtrain.append((t[0],t[1] + 1,t[2]))
        self.new_train = newtrain
        print("RELATION+1")
    def preproc(self):
        segtrain = {i+1:[] for i in range(self.rnum)}
        ht_r = {i: {} for i in range(self.enum )}
        hr_t = {}
        for number, t in enumerate(self.new_train) :
            self.graph[t[0]].add(t[2])
            self.graph[t[2]].add(t[0])
            if t[2] not in ht_r[t[0]].keys():
                ht_r[t[0]][t[2]] = []
            if t[0] not in ht_r[t[2]].keys():
                ht_r[t[2]][t[0]] = []
            ht_r[t[0]][t[2]].append(t[1] )
            ht_r[t[2]][t[0]].append(-t[1])
            if (t[0], t[1]) not in hr_t:
                hr_t[(t[0], t[1])] = set()
            if (t[2], -t[1]) not in hr_t:
                hr_t[(t[2], -t[1])] = set()
            hr_t[(t[0], t[1] )].add(t[2])
            hr_t[(t[2], -t[1])].add(t[0])
            segtrain[t[1]].append(t)
        self.hr_t = hr_t
        return segtrain,hr_t,ht_r
    def Convertpath(self,p,relati):
        if len(p) <= 2:
            return
        coe = 1
        tup = ()
        pathlen = len(p) - 1
        for ll in range(pathlen):
            r = self.Chos(self.ht_r[p[ll]][p[ll + 1]],p[ll])
            tup = tup + (r,)
            if (p[ll], r) in self.hr_t:
                temp = (p[ll], r)
                coe = coe * len(self.hr_t[temp])
        if tup not in self.Rules[relati]:
            self.Rules[relati][tup] = 0
        self.Rules[relati][tup] += 1 / coe
        # self.Rules[relati][tup] += 1 / len(tup)
    def find1path(self,subt):
        for t in subt:
            for r in self.ht_r[t[0]][t[2]]:
                if r != (t[1]):
                    if (r,) not in self.Rules[t[1]]:
                        self.Rules[t[1]][(r,)] = 0
                    coe = len(self.hr_t[(t[0],t[1])] & self.hr_t[(t[0],r)])
                    self.Rules[t[1]][(r,)] += 1 / len(self.hr_t[(t[0], r)])*coe
                    # self.Rules[t[1]][(r,)] += 1

        return
    def bidirectional_bfs(self,graph, start):
        answer_num = len(self.hr_t[(start, self.current[1])])
        answer = self.hr_t[(start, self.current[1])]
        if answer_num == 1 and set().add(start) == answer:
            return [[start]]
        forward_queue = deque([(start, [start])])
        backward_queue = deque([])
        forward_visited = defaultdict(list)
        backward_visited = defaultdict(list)
        forward_visited[start].append([start])
        backward_visited[self.current[2]].append([self.current[2]])
        backward_queue.append((self.current[2], [self.current[2]]))
        #Q = 5
        Answer =  copy.deepcopy(answer)
        if answer_num > self.max_answer:
            Answer = random.sample(answer,self.max_answer)
        for ans in Answer:
            if ans != self.current[2]:
                backward_visited[ans].append([ans])
                # backward_queue.append((ans, [ans]))
        X = math.ceil(self.pathlenth/2)
        for step in range(X):
            # Expand forward
            self.expand(graph, forward_queue, forward_visited, backward_visited,  direction='forward')
            # Expand backward
            if (self.pathlenth%2 == 1 and step<(X-1)) or self.pathlenth%2==0:
                self.expand(graph, backward_queue, backward_visited, forward_visited,  direction='backward')
    def expand(self,graph, queue, current_visited, other_visited,  direction='forward'):
        level_size = len(queue)
        for _ in range(level_size):
            current_node, path = queue.popleft()
            WAITforexpand = graph[current_node]
            if len(WAITforexpand) > self.maxnei:
                WAITforexpand = random.sample(WAITforexpand,self.maxnei)
            for neighbor in WAITforexpand:
                if neighbor in path:
                    continue  # Avoid cycles
                new_path = path + [neighbor]
                if direction == 'forward':
                    if neighbor in other_visited:
                        for other_path in other_visited[neighbor]:
                            if self.has_common_elements(new_path,other_path[::-1][1:]):
                                continue
                            combined_path = new_path + other_path[::-1][1:]
                            if len(combined_path) <= (self.pathlenth + 1):
                                self.Convertpath(combined_path,self.current[1])
                else:
                    if neighbor in other_visited:
                        for other_path in other_visited[neighbor]:
                            if self.has_common_elements(new_path[::-1][1:],other_path):
                                continue
                            combined_path = other_path + new_path[::-1][1:]
                            if len(combined_path) <= (self.pathlenth+1):
                                self.Convertpath(combined_path,self.current[1])
                current_visited[neighbor].append(new_path)
                queue.append((neighbor, new_path))
        return
    def save_rule(self, Dataname):
        Dataname = Dataname
        result_dir = Path("Rule") / Dataname
        result_dir.mkdir(parents=True, exist_ok=True)
        rule_name = f"{self.pathlenth}_{self.samplenum}_{self.maxnei}result"
        rule_path = result_dir / rule_name
        try:
            with rule_path.open("wb") as f:
                pickle.dump(self.Rules, f)
            print(f"Rule Save to: {rule_path}")
        except Exception as e:
            print(f"Error occur when saving: {e}")

    def train(self):
        for rindex in range(1,self.rnum+1):
            subt = self.sample_facts(self.segtrain[rindex])
            print("------------------------Relation:{0}  facts:{1}------------------------".format(rindex, len(subt)))
            self.find1path(subt)
            for index,t in enumerate(subt):
                self.current = t
                start_node = t[0]
                self.graph[t[0]].discard(t[2])
                self.graph[t[2]].discard(t[0])
                self.bidirectional_bfs(self.graph, start_node)
                self.graph[t[0]].add(t[2])
                self.graph[t[2]].add(t[0])
                if index%100==0:
                    print("Completion:", round(index / len(subt),3))
            #Normalization
            for rule,conf in self.Rules[rindex].items():
                self.Rules[rindex][rule] = conf/len(subt)
            self.Rules[rindex] = self.get_top_items(self.Rules[rindex],1000)
        self.save_rule(Dataname)

@contextmanager
def time_block(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{label}] cost: {elapsed:.4f}seconds")

if __name__ == "__main__":
    random.seed(10)
    Dataname = "YAGO3-10"      # 6,300,100
    # Dataname = "NELL-995"  # 3,300,100
    # Dataname = "YAGO3-10"  # 3,300,100
    # Dataname = "FB15K-237" # 3,300,100
    trainlist, testlist, entity_dict, relation_dict = Getdata(Dataname)
    M = MPRM(trainlist,len(entity_dict),len(relation_dict),3,300,maxnei=100)
    with time_block("train"):
        M.train()





