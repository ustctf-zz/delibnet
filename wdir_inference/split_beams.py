# coding=utf-8

import sys

try:
    tok = "\t"#sys.argv[2]
except:
    tok = r'<-beam->'


all_lines = []
with open(sys.argv[1], "r") as f:
    map(lambda x: all_lines.extend(x.split(tok)), f)

with open(sys.argv[1] + '.beamouts', "w") as ff:
    for x in all_lines:
        print >>ff, x.strip()
