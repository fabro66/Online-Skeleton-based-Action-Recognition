import numpy as np


setup_file = 'setup.txt'
skes_name_file = 'skes_available_name.txt'
skes_names = np.loadtxt(skes_name_file, dtype=np.string_)

setups = []
for name in skes_names:
    setup_num = int(name[1:4])
    setups.append(setup_num)

setups = np.asarray(setups, dtype=np.int)
np.savetxt(setup_file, setups, fmt='%d')


performer_file = 'performer.txt'
performers = []
for name in skes_names:
    performer_num = int(name[9:12])
    performers.append(performer_num)

performers = np.asarray(performers, dtype=np.int)
np.savetxt(performer_file, performers, fmt="%d")


label_file = 'label.txt'
labels = []
for name in skes_names:
    label_num = int(name[-3:])
    labels.append(label_num)

labels = np.asarray(labels, dtype=np.int)
np.savetxt(label_file, labels, fmt="%d")