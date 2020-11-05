# https://github.com/shahroudy/NTURGB-D
ntu_pairs_ori = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                 (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                 (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23),
                 (21, 21), (23, 8), (24, 25), (25, 12))
ntu_pairs = [(v1 - 1, v2 - 1) for v1, v2 in ntu_pairs_ori]

ntu_symmetry_ori = ((5, 9), (6, 10), (7, 11), (8, 12), (22, 24), (23, 25), (13, 17), (14, 18), (15, 19), (16, 20))
ntu_symmetry = [(i - 1, j - 1) for i, j in ntu_symmetry_ori]

GAST_paris = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5),
              (7, 0), (8, 7), (9, 8), (10, 9), (11, 8), (12, 11),
              (13, 12), (14, 8), (15, 14), (16, 15)]
GAST_symmetry = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]


class Skeleton:
    def __init__(self, mode='NTU'):
        assert mode in ['NTU60', 'NTU120', 'GAST60', 'GAST120']
        if mode.startswith('NTU'):
            self.mode = 'ntu'
        elif mode.startswith('GAST') or mode.startswith('HRN'):
            self.mode = 'gast'

        # The order of joints in NTU60/120
        self._ntu_pairs = ntu_pairs
        self._ntu_symmetry = ntu_symmetry
        self._ntu_num_joints = 25
        self._ntu_limb_blocks = [[9, 10, 11, 8, 24, 23], [5, 6, 7, 4, 22, 21],
                                 [17, 18, 19, 16], [13, 14, 15, 12],
                                 [0, 1, 2, 3, 20]]  # left arm, right arm, left leg, right leg, head spine

        # The order of joints in NTU-GAST skeleton
        self._gast_pairs = GAST_paris
        self._gast_symmetry = GAST_symmetry
        self._gast_num_joints = 17
        self._gast_limb_blocks = [[14, 15, 16], [11, 12, 13], [1, 2, 3], [4, 5, 6], [0, 7, 8, 9, 10]]

    def get_pairs(self):
        return eval('self._{}_pairs'.format(self.mode))

    def get_symmetry(self):
        return eval('self._{}_symmetry'.format(self.mode))

    def get_num_joints(self):
        return eval('self._{}_num_joints'.format(self.mode))

    def get_limb_blocks(self):
        """
        return: [left_arm, right_arm, left_leg, right_leg, head_spine]
        """
        return eval('self._{}_limb_blocks'.format(self.mode))
