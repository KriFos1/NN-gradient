import csv
import numpy as np
import pickle
import pandas as pd

data_index = [('6kHz','83ft'),('12kHz','83ft'),('24kHz','83ft'),('24kHz','43ft'),('48kHz','43ft'),('96kHz','43ft')]
datatyp = 'Bfield'

path_to_benchmark = 'data/Benchmark-3/'

T=np.loadtxt(f'{path_to_benchmark}ascii/trajectory.DAT',comments='%')
tvd =T[:,1]
#ED = T[:,5]

#with open('../inputs/trajectory.DAT','r') as f:
#    lines = f.readlines()
#    tvd = [el.strip().split()[1] for el in lines][1:]

k = open('assim_index.csv','w',newline='')
#writer4 = csv.writer(k)
l = open('datatyp.csv','w',newline='')
writer5 = csv.writer(l)

# build a pandas dataframe with the data.
# The tvd is the index and the tuple (freq,dist) is the columns


data = {}
var = {}
for di in data_index:
    freq, dist = di
    try:
        with open(f'{path_to_benchmark}/logdata/{datatyp}_{dist}_{freq}.las', 'r') as f:
            lines = f.readlines()
            values = [np.array(el.strip().split()[1:],dtype=np.float32) for el in lines[1:]]
            data[(freq, dist)] = values
            #var[(freq, dist)] = [[['REL', 10] if abs(el) > abs(0.1*np.mean(values)) else ['ABS', (0.1*np.mean(values))**2] for el in val] for val in values]
            var[(freq, dist)] = [['ABS'] + [[(0.02*np.mean(values))**2 for el in val]] for val in values]
    except:
        data[(freq, dist)] = None
        var[(freq,dist)] = None
        pass

df = pd.DataFrame(data,columns=data_index,index=tvd)
df.index.name = 'tvd'
#df.to_csv('data.csv',index=True)
df.to_pickle('data.pkl')

df = pd.DataFrame(var,columns=data_index,index=tvd)
df.index.name = 'tvd'
df.to_csv('var.csv',index=True)
with open('var.pkl','wb') as f:
    pickle.dump(df,f)

#filt = [i*10 for i in range(50)]
for c,_ in enumerate(tvd):
    #if c in filt:
    k.writelines(str(c) + '\n')
k.close()

writer5.writerow([str(el) for el in data_index])
l.close()
