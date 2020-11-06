import os
import numpy as np
import h5py
import pickle as pkl
from tqdm import tqdm


root_path = './data/NTU-GAST-Skeleton'
label_file = './data/gast120/statistics/label.txt'
camera_file = './data/gast120/statistics/camera.txt'
performer_file = './data/gast120/statistics/performer.txt'
skes_name_file = './data/gast120/statistics/skes_available_name.txt'
skes_pkl_file = './data/gast120/statistics/skes_gast.pkl'

dicts = ['S{:02d}'.format(i) for i in range(1, 17)]

labels = []
cameras = []
performers = []
available_names = []
skeletons_available = []

for dict in dicts:
    print('Processing %s' % dict)
    skes_names = os.listdir(os.path.join(root_path, dict))
    skes_names.sort()

    for skes_name in tqdm(skes_names):
        skes_path = os.path.join(root_path, dict, skes_name)

        with h5py.File(skes_path, 'r') as fr:
            skeletons = fr['skeletons'][()]

        valid_frames = np.where(np.sum(skeletons.reshape(-1, 51), axis=1) != 0)[0]
        num_valid_frames = len(valid_frames)

        if num_valid_frames >= 20:
            cameras += [int(skes_name[5:8])]
            performers += [int(skes_name[9:12])]
            labels += [int(skes_name[17:20])]
            available_names += ['{}\n'.format(skes_name[:-3])]

            num_body = skeletons.shape[0]
            if num_body == 1:
                # skeletons = skeletons.squeeze()
                skeletons = skeletons.reshape(-1, 51)
            else:
                skeletons = skeletons.transpose(1, 0, 2, 3).reshape(-1, 102)

            skeletons_available.append(skeletons)

print('Saving file ...')
cameras = np.array(cameras, dtype=np.int)
performers = np.array(performers, dtype=np.int)
labels = np.array(labels, dtype=np.int)

np.savetxt(camera_file, cameras, fmt='%d')
np.savetxt(performer_file, performers, fmt='%d')
np.savetxt(label_file, labels, fmt='%d')

with open(skes_name_file, 'w') as fw:
    for available_name in available_names:
        fw.write(available_name)


with open(skes_pkl_file, 'wb') as fw:
    pkl.dump(skeletons_available, fw, pkl.HIGHEST_PROTOCOL)

print('Finishing!')
