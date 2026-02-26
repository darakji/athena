# partition_cifs.py
import csv

NGPU = 4
groups = [[] for _ in range(NGPU)]
loads = [0]*NGPU

with open("/home/mehuldarak/athena/polaris/scripts/atom_counts.csv") as f:
    rows = [(r[0], int(r[1])) for r in csv.reader(f)]

for name, nat in sorted(rows, key=lambda x: -x[1]):
    i = loads.index(min(loads))
    groups[i].append(name)
    loads[i] += nat

for i,g in enumerate(groups):
    with open(f"group_gpu{i}.txt", "w") as f:
        f.write("\n".join(g))