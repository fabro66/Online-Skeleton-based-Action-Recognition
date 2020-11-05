import os


def make_dir(dataset):
    if dataset == 'NTU60':
        output_dir = os.path.join('./results/NTU60/')
    elif dataset == 'NTU120':
        output_dir = os.path.join('./results/NTU120/')
    elif dataset == 'GAST60':
        output_dir = os.path.join('./results/GAST60/')
    elif dataset == 'GAST120':
        output_dir = os.path.join('./results/GAST120/')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir


def get_num_classes(dataset):
    if dataset in ['NTU60', 'GAST60']:
        return 60
    elif dataset in ['NTU120', 'GAST120']:
        return 120
