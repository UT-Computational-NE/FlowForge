import re
import numpy as np
def replace_string(substr, mydict):
    idx=0
    splt=substr[1:-1].split("%")
    fmtstr=False
    if(len(splt)==1):
        newsubstr=splt[0]
    elif(len(splt)==2):
        newsubstr=splt[0]
        fmtstr="{:"+splt[1]+"}"
    else:
        raise RuntimeError('Invalid string found: '+substr)
    splt=newsubstr.split(":")
    if(len(splt)==1):
        pass
    elif(len(splt)==2):
        idx=int(splt[1])
    else:
        raise RuntimeError('Invalid string found: '+substr)
    if not splt[0] in mydict:
        print(mydict)
        raise RuntimeError('Invalid key found: '+splt[0])
    if(isinstance(mydict[splt[0]], (list, np.ndarray))):
        if len(mydict[splt[0]]) - 1 < idx:
            raise RuntimeError('Exceeded array length: '+substr)
        if(not fmtstr):
            return str(mydict[splt[0]][idx])
        else:
            try:
                return fmtstr.format(mydict[splt[0]][idx])
            except:
                raise RuntimeError('Bad format string: '+fmtstr)
    else:
        if(not fmtstr):
            return str(mydict[splt[0]])
        else:
            try:
                return fmtstr.format(mydict[splt[0]])
            except:
                raise RuntimeError('Bad format string: '+fmtstr)

def replace_line(line, mydict):
    p = re.compile(r"\{[^}]+\}")
    outline=line
    for match in p.finditer(outline):
        outline = outline.replace(match.group(),replace_string(match.group(), mydict))
    return outline

def replace(template, target, mydict):
    with open(template) as fin, open(target,'w+') as fout:
        for line in fin:
            fout.write(replace_line(line, mydict))

if __name__=='__main__':
    import json
    import argparse

    parser = argparse.ArgumentParser(description='Process input to template.')
    parser.add_argument("-t", type=str, help='template file')
    parser.add_argument("-d", type=str, help='destination file')
    parser.add_argument('-j', type=str, help='JSON file to feed into template')

    args = parser.parse_args()

    with open(args.j) as json_file:
        mydict = json.load(json_file)

    replace(args.t, args.d, mydict)