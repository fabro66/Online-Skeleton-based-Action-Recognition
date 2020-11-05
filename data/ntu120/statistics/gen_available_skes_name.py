import numpy as np
import os
from tqdm import tqdm


miss_names = './samples_with_missing_skeletons.txt'
skes_path = '/home/data/junfa/NTU_Skeleton/NTU120'
availabel_skes_name = './skes_available_name.txt'

miss_names = np.loadtxt(miss_names, dtype=str)
skes_names = os.listdir(skes_path)

skes_names.sort()
available_skes = []
for name in tqdm(skes_names):
    name = name.split('.')[0]
    if name not in miss_names:
        available_skes.append(name)

with open(availabel_skes_name, 'w') as fw:
    for name in available_skes:
        name = name + '\n'
        fw.write(name)

print('Finishing~')
