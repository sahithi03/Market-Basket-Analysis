from pyspark import SparkContext
import sys
import time
from collections import defaultdict
from collections import Counter
import itertools

sc = SparkContext('local[*]','task2')
sc.setLogLevel("ERROR")

start = time.time()

filter_threshold = int(sys.argv[1])
support = int(sys.argv[2])
input_file = sys.argv[3]
output_file_path = sys.argv[4]
processed_filepath = "Customer_product.csv"

input_data = sc.textFile(input_file)

# ************ PRE PROCESS DATA ***************************
header1 = input_data.first()
required_data = input_data.filter(lambda l:l != header1).map(lambda x:x.replace('"','').split(",")).map(lambda x:(("-".join([x[0],x[1]])),x[5].lstrip('0'))).collect()

def create_csv(required_data,processed_filepath):
    with open(processed_filepath,"w+") as pf:
        pf.write("DATE-CUSTOMER_ID,PRODUCT_ID\n")
        for row in required_data:
            pf.write(row[0] + "," + row[1]+"\n")
    pf.close()

create_csv(required_data,processed_filepath)

date_cust_rdd = sc.textFile(processed_filepath)
header = date_cust_rdd.first()
baskets = date_cust_rdd.filter(lambda l: l != header).map(lambda x:x.split(",")).map(lambda x:(x[0],[x[1]])).reduceByKey(lambda x,y:x+y).filter(lambda x:len(x[1]) > filter_threshold).mapValues(set)
all_baskets = baskets.map(lambda x:x[1])

total_count = all_baskets.count()

# ************* APRIORI ALGORITHM *******************

def get_candidates(frequent_items,k):
    candidates = []
    for i in range(len(frequent_items)-1):
        for j in range(i+1,len(frequent_items)):
            if (frequent_items[i][:k-2] == frequent_items[j][:k-2]):
                candidate = list(set(frequent_items[i]).union(set(frequent_items[j])))
                candidate.sort()
                candidates.append(candidate)
            else:
                break
    return candidates

def get_frequent_items(candidates,support,baskets):
    freq_dictionary = defaultdict(int)
    frequent_items = []
    for candidate in candidates:
        key = tuple(candidate)
        for basket in baskets:
            if set(candidate).issubset(basket):
                freq_dictionary[key] += 1
    for key,value in freq_dictionary.items():
        if value >= support:
            frequent_items.append(key)
    return frequent_items

def apriori(baskets,support,total_count):
    baskets = list(baskets)
    final_result = []
    threshold = support * (float(len(baskets)) / float(total_count))

    C1 = Counter()
    for basket in baskets:
        C1.update(basket)

    L = []
    for key,value in C1.items():
        if value >= threshold:
            L.append(key)
    L1 = [(x,) for x in L]
    final_result.extend(L1)

    k = 2
    frequent_items = set(L)
    while True:
        if k == 2:
            C2 = []
            for x in itertools.combinations(frequent_items, 2):
                pair = list(x)
                pair.sort()
                C2.append(pair)
            candidates = C2
        else:
            candidates = get_candidates(frequent_items,k)
        next_frequent_items = get_frequent_items(candidates,threshold,baskets)
        final_result.extend(next_frequent_items)
        frequent_items = list(set(next_frequent_items))
        frequent_items.sort()
        k += 1
        if len(frequent_items) == 0:
            break
    return final_result

# *********** SON ALGORITHM ************************
def SONPhase2_count_candidates(baskets,candidates):

    item_counts = defaultdict(int)
    baskets = list(baskets)
    for candidate in candidates:
        key = candidate
        for basket in baskets:
            if set(candidate).issubset(basket):
                item_counts[key] += 1

    return item_counts.items()

#--------------PHASE 1----------------------------
SONPhase1_map = all_baskets.mapPartitions(lambda baskets: apriori(baskets,support,total_count)).map(lambda x:(x,1))
SONPhase1_reduce = SONPhase1_map.reduceByKey(lambda x,y:x+y).map(lambda x:x[0]).collect()

candidates_list = sorted(SONPhase1_reduce,key = lambda x:(len(x),x))

#------------PHASE 2 -------------------------------
SONPhase2_map = all_baskets.mapPartitions(lambda baskets: SONPhase2_count_candidates(baskets,SONPhase1_reduce))
SONPhase2_reduce = SONPhase2_map.reduceByKey(lambda x,y: (x+y)).filter(lambda x:x[1]>=support).map(lambda x:x[0]).collect()


frequent_itemsets = sorted(SONPhase2_reduce,key = lambda x:(len(x),x))


outfile = open(output_file_path,'w')
outfile.write("Candidates:")
outfile.write("\n")
if len(candidates_list) != 0:
    prev_length = len(candidates_list[0])
    first_val = str(candidates_list[0]).replace(',','')
    outfile.write(first_val)
    for i in range(1,len(candidates_list)):
        curr_length = len(candidates_list[i])
        if prev_length == curr_length:
            outfile.write(",")
        else:
            outfile.write("\n\n")
        if curr_length == 1:
            val = str(candidates_list[i]).replace(',','')
        else:
            val = str(candidates_list[i])
        outfile.write(val)
        prev_length = curr_length
outfile.write("\n\n")
outfile.write("Frequent Itemsets:")
outfile.write("\n")
if len(frequent_itemsets) != 0:
    prev_length = len(frequent_itemsets[0])
    first_val = str(frequent_itemsets[0]).replace(',','')
    outfile.write(first_val)
    for i in range(1,len(frequent_itemsets)):
        curr_length = len(frequent_itemsets[i])
        if curr_length == 1:
            val = str(frequent_itemsets[i]).replace(',','')
        else:
            val = str(frequent_itemsets[i])
        if prev_length == curr_length:
            outfile.write(",")
        else:
            outfile.write("\n\n")
        outfile.write(val)
        prev_length = curr_length
outfile.close()

print("Duration:",time.time()-start)