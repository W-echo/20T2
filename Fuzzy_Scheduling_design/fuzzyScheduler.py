# assignment 1 of COMP9414 python code
# authored: Wang Xiaoyu, ID: z5190937

import sys
import searchGeneric
from searchProblem import *
from cspProblem import Constraint, CSP
import cspConsistency
from cspExamples import *



Date = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}
Time = {'9am': 9, '10am': 10, '11am': 11, '12pm': 12, '1pm': 13, '2pm': 14, '3pm': 15, '4pm': 16, '5pm': 17}
# reserve cost num for no pass way
MAX_cost = 9999

# produce and return possible values for each task
def value_domain(x):
    """ x is the duration hour of a task """
    # task must be finished in one day
    # d is to store possible values of tasks
    d = set()
    for i in range(1, len(Date)+1):
        for j in range(0, len(Time)-x):
            # start time for first(), and end time second()
            d.add(((i, 9+j), (i, 9+j+x)))
    return d


# get the key in a dic according to its value
def get_key(dict, value):
    for k, v in dict.items():
        if v == value:
            return k


#            binary constrains          #
def before(a, b):
    t1_start, t1_end = a
    t2_start, t2_end = b
    t1_date, t1_time = t1_end
    t2_date, t2_time = t2_start
    if t1_date < t2_date:
        return True
    elif t1_date == t2_date and t1_time <= t2_time:
        return True
    else:
        return False


def after(a, b):
    t1_start, t1_end = a
    t2_start, t2_end = b
    t1_date, t1_time = t1_start
    t2_date, t2_time = t2_end
    if t2_date < t1_date:
        return True
    elif t1_date == t2_date and t2_time < t1_time:
        return True
    else:
        return False


def same_day(a, b):
    t1_start, t1_end = a
    t2_start, t2_end = b
    t1_date, t1_time = t1_start
    t2_date, t2_time = t2_start
    if t2_date == t1_date:
        return True
    else:
        return False


def start_at(a, b):
    t1_start, t1_end = a
    t2_start, t2_end = b
    t1_date, t1_time = t1_start
    t2_date, t2_time = t2_end
    if t2_date == t1_date and t2_time == t1_time:
        return True
    else:
        return False


#            hard domain constraints         #
def at_(val):
    # val = Time[frame]

    def nev(x):
        start, end = x
        x_date, x_time = start
        return val == x_time
    nev.__name__ = "at " + str(val)     # name of the function
    return nev


def on_(val):
    # val = Date[frame]

    def nev(x):
        start, end = x
        x_date, x_time = start
        return val == x_date
    nev.__name__ = "on " + str(val)      # name of the function
    return nev


def start_before_(frame):
    date, time = frame

    def bqv(x):
        start, end = x
        x_date, x_time = start
        return x_date<date or (x_date==date and x_time <= time)
    bqv.__name__ = "starts before" + str(frame)
    return bqv


def start_after_(frame):
    date, time = frame

    def aqv(x):
        start, end = x
        x_date, x_time = start
        return x_date > date or (x_date == date and x_time >= time)
    aqv.__name__ = "start after" + str(frame)
    return aqv


def ends_before_(frame):
    date, time = frame

    def ends_before(x):
        start, end = x
        x_date, x_time = end
        return x_date < date or (x_date == date and x_time <= time)
    ends_before.__name__ = " ends before" + str(frame)
    return ends_before


def ends_after_(frame):
    date, time = frame

    def ends_after(x):
        start, end = x
        x_date, x_time = end
        return x_date > date or (x_date == date and x_time >= time)
    ends_after.__name__ = "ends after" + str(frame)
    return ends_after


def start_in_(a, b):
    begin_date, begin_time = a
    final_date, final_time = b

    def start_in(x):
        start, end = x
        x_date, x_time = start
        return (x_date > begin_date or (x_date == x_date and x_time >= begin_time)) \
               and (x_date < final_date or (x_date == final_date and x_time <= final_time))
    start_in.__name__ = "starts between " + str(a) + "and" + str(b)
    return start_in


def ends_in_(a, b):
    begin_date, begin_time = a
    final_date, final_time = b

    def ends_in(x):
        start, end = x
        x_date, x_time = end
        return (x_date > begin_date or (x_date == x_date and x_time >= begin_time)) \
               and (x_date < final_date or (x_date == final_date and x_time <= final_time))

    ends_in.__name__ = "ends between " + str(a) + "and" + str(b)
    return ends_in


def start_before_time_(frame):
    def start_before_time(x):
        start, end = x
        x_date, x_time = start
        return x_time <= frame
    start_before_time.__name__ = "start_before_time" + str(frame)
    return start_before_time


def end_before_time_(frame):
    def end_before_time(x):
        start, end = x
        x_date, x_time = end
        return x_time <= frame
    end_before_time.__name__ = "end_before_time" + str(frame)
    return end_before_time


def start_after_time_(frame):
    def start_after_time(x):
        start, end = x
        x_date, x_time = start
        return x_time >= frame
    start_after_time.__name__ = "start_after_time" + str(frame)
    return start_after_time


def end_after_time_(frame):
    def end_after_time(x):
        start, end = x
        x_date, x_time = end
        return x_time >= frame
    end_after_time.__name__ = "end_after_time" + str(frame)
    return end_after_time


#           soft deadline constraints       #
def ends_by_(frame):
    date, time = frame

    def ends_before(x):
        start, end = x
        x_date, x_time = end
        return x_date < date or (x_date == date and x_time <= time)

    ends_before.__name__ = "ends by <=" + str(frame)
    return ends_before


class Extendedcsp(CSP):
    def __init__(self, domains, constraints, soft_constraints):
        super().__init__(domains, constraints)
        self.soft_constraints = soft_constraints



class Search_with_AC_from_cost_CSP(cspConsistency.Search_with_AC_from_CSP):
    def __init__(self, csp):
        super().__init__(csp)
        self.cost_list = []

    def sub_softcost(self, a, b):
        act_date, act_time = a
        idea_date, idea_time = b
        if act_date < idea_date or (act_date == idea_date and act_time <= idea_time):
            return 0
        else:
            if act_date == idea_date:
                return act_time - idea_time
            else:
                return (act_date - idea_date)*24 + act_time - idea_time


    def heuristic(self, n):
        """Gives the heuristic value of node n.
        Returns 0 if not overridden. (cost_sum)"""
        # print(n)
        cost_sum = 0
        cost = {}
        soft_c = csp.soft_constraints
        # print('soft:', soft_c)
        for item in soft_c:
            t = item[0]
            deadline = item[1]
            temp_cost = []
            for act_frame in n[t]:
                start, end = act_frame
                temp_cost.append(int(item[-1])*self.sub_softcost(end, deadline))
            if temp_cost==[]:
                cost[t] = MAX_cost
            else:
                cost[t] = min(temp_cost)
        for c in cost:
            cost_sum += cost[c]
        self.cost_list.append(cost_sum)
        return cost_sum


# used for store task, hard constraints and soft constraints
task = {}
cons = []
soft_cons = []
cost_unit = {}

# get the input file name
file_name = sys.argv[1]
# file_name = 'test.txt'
temp_list = []

#read txt
f = open(file_name,'r')
lines = f.readlines()
for line in lines:
    line = line.replace("\n", "")
    if "#" not in line:
        temp_list.append(line.split(" "))
    # print(l)
f.close()
# print(temp_list)
# print(value_domain(3))

# deal with valid line and distribute them into task, all constraints and soft constraints
for line in temp_list:
    # print(line)
    if "task," in line:
        # print(line[1])
        task[line[1]] = value_domain(int(line[-1]))
    # soft constraints
    elif "ends-by" in line:
        t = (Date[line[3]], Time[line[4]])
        cost_unit[line[1]] = line[-1]
        soft_cons.append([line[1], t, line[-1]])
    # binary constraints of cons
    elif "constraint," in line:
        if "before" in line:
            cons.append(Constraint((line[1], line[3]), before))
        elif "after" in line:
            cons.append(Constraint((line[1], line[3]), after))
        elif "same-day" in line:
            cons.append(Constraint((line[1], line[3]), same_day))
        elif "starts-at" in line:
            cons.append(Constraint((line[1], line[3]), start_at))
    # hard domain constraints
    elif "domain," in line:
        if len(line) == 3 and line[-1] in Date:
            date = Date[line[-1]]
            cons.append(Constraint((line[1], ), on_(date)))
        elif len(line) == 3 and line[-1] in Time:
            time = Time[line[-1]]
            cons.append(Constraint((line[1], ), at_(time)))
        elif "starts-before" in line and len(line) == 5:
            date_time = (Date[line[-2]], Time[line[-1]])
            cons.append(Constraint((line[1],), start_before_(date_time)))
        elif "starts-after" in line and len(line) == 5:
            date_time = (Date[line[-2]], Time[line[-1]])
            cons.append(Constraint((line[1],), start_after_(date_time)))
        elif "ends-before" in line and len(line) == 5:
            date_time = (Date[line[-2]], Time[line[-1]])
            cons.append(Constraint((line[1],), ends_before_(date_time)))
        elif "ends-after" in line and len(line) == 5:
            date_time = (Date[line[-2]], Time[line[-1]])
            cons.append(Constraint((line[1],), ends_after_(date_time)))
        elif "starts-in" in line:
            num = line[4].index('-')
            date_time1 = (Date[line[3]], Time[line[4][:num]])
            date_time2 = (Date[line[4][num+1:]], Time[line[-1]])
            cons.append(Constraint((line[1],), start_in_(date_time1, date_time2)))
        elif "ends-in" in line:
            num = line[4].index('-')
            date_time1 = (Date[line[3]], Time[line[4][:num]])
            date_time2 = (Date[line[4][num+1:]], Time[line[-1]])
            cons.append(Constraint((line[1],), ends_in_(date_time1, date_time2)))
        elif "starts-before" in line and len(line) == 4:
            time = Time[line[-1]]
            cons.append(Constraint((line[1],), start_before_time_(time)))
        elif "starts-after" in line and len(line) == 4:
            time = Time[line[-1]]
            cons.append(Constraint((line[1],), start_after_time_(time)))
        elif "ends-before" in line and len(line) == 4:
            time = Time[line[-1]]
            cons.append(Constraint((line[1],), end_before_time_(time)))
        elif "ends-after" in line and len(line) == 4:
            time = Time[line[-1]]
            cons.append(Constraint((line[1],), end_after_time_(time)))

# print(task)
# print(cons)
# print(soft_cons)
# print(cost_unit)

# get the csp
csp = Extendedcsp(task, cons, soft_cons)
# print(csp)
# search for path
searcher = searchGeneric.AStarSearcher(Search_with_AC_from_cost_CSP(csp))
result = searcher.search()
# print("-----------")


if result and min(searcher.problem.cost_list)<= MAX_cost:
    # print(solution and cost)
    ans = {m: cspConsistency.select(n) for (m, n) in result.end().items()}
    for x in ans:
        start, _ = ans[x]
        date, time = start
        print(f'{x}:{get_key(Date,date)} {get_key(Time,time)}')

    print(f'cost:{min(searcher.problem.cost_list)}')
else:
    print('No solution')