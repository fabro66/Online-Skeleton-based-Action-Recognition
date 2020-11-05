from torch import nn
import torch
import numpy as np
import math
from tools.graph import get_adj_sym_matrix


class ESN(nn.Module):
    def __init__(self, num_classes, skeleton, seg, bias=True):
        super(ESN, self).__init__()
        self.limb_blocks = skeleton.get_limb_blocks()
        self.pairs = skeleton.get_pairs()
        self.num_joints = skeleton.get_num_joints()

        adj_graph, sym_graph = get_adj_sym_matrix(skeleton, norm=True)
        adj_graph = torch.tensor(adj_graph, dtype=torch.float32, requires_grad=False)
        sym_graph = torch.tensor(sym_graph, dtype=torch.float32, requires_grad=False)

        self.register_buffer('adj_graph', adj_graph)
        self.register_buffer('sym_graph', sym_graph)

        self.dim1 = 256
        self.seg = seg

        self.m_adj = adj_graph > 0
        self.e_adj = nn.Parameter(adj_graph[self.m_adj])
        self.m_sym = sym_graph > 0
        self.e_sym = nn.Parameter(sym_graph[self.m_sym])

        self.joint_embed = embed(3, 64, self.num_joints, norm=True, bias=bias)
        self.joint_dif_embed = embed(3, 64, self.num_joints, norm=True, bias=bias)
        self.bone_embed = embed(3, 64, self.num_joints, norm=True, bias=bias)
        self.bone_dif_embed = embed(3, 64, self.num_joints, norm=True, bias=bias)
        self.conv_cat = nn.Conv2d(64*4, 128, 1, bias=bias)
        self.bn_cat = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.limb_blocks, self.dim1, 2 * self.dim1, bias=bias)
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)

        self.fc = nn.Linear(self.dim1 * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

    def forward(self, input):
        # Dynamic Representation
        bs, step, dim = input.size()
        num_joints = dim // 3
        input = input.view((bs, step, num_joints, 3))
        input = input.permute(0, 3, 2, 1).contiguous()  # (B, C, N, T)

        joint_dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        joint_dif = torch.cat([joint_dif.new(bs, joint_dif.size(1), num_joints, 1).zero_(), joint_dif], dim=-1)

        bone = torch.zeros_like(input)
        for v1, v2 in self.pairs:
            bone[:, :, v1, :] = input[:, :, v1, :] - input[:, :, v2, :]
        bone_dif = bone[:, :, :, 1:] - bone[:, :, :, 0:-1]
        bone_dif = torch.cat([bone_dif.new(bs, bone_dif.size(1), num_joints, 1).zero_(), bone_dif], dim=-1)

        joint = self.joint_embed(input)
        joint_dif = self.joint_dif_embed(joint_dif)
        bone = self.bone_embed(bone)
        bone_dif = self.bone_dif_embed(bone_dif)

        # Combine
        input = torch.cat((joint_dif, joint, bone_dif, bone), dim=1)
        input = self.relu(self.bn_cat(self.conv_cat(input)))

        adj = torch.zeros_like(self.adj_graph)
        adj[self.m_adj] = self.e_adj
        sym = torch.zeros_like(self.sym_graph)
        sym[self.m_sym] = self.e_sym
        g = self.compute_g1(input)
        g = g + adj + sym
        input = self.gcn1(input, g)
        input = self.gcn2(input, g)
        input = self.gcn3(input, g)

        input = self.cnn(input)
        # Classification
        output = self.maxpool(input)
        output = torch.flatten(output, 1)
        output = self.fc(output)

        return output


class ESNKinetics(nn.Module):
    def __init__(self, num_classes, skeleton, seg, bias=True):
        super(ESNKinetics, self).__init__()
        self.limb_blocks = skeleton.get_limb_blocks()
        self.pairs = skeleton.get_pairs()
        self.num_joints = skeleton.get_num_joints()

        adj_graph, sym_graph = get_adj_sym_matrix(skeleton, norm=True)
        adj_graph = torch.tensor(adj_graph, dtype=torch.float32, requires_grad=False)
        sym_graph = torch.tensor(sym_graph, dtype=torch.float32, requires_grad=False)

        self.register_buffer('adj_graph', adj_graph)
        self.register_buffer('sym_graph', sym_graph)

        self.dim1 = 256
        self.seg = seg

        self.m_adj = adj_graph > 0
        self.e_adj = nn.Parameter(adj_graph[self.m_adj])
        self.m_sym = sym_graph > 0
        self.e_sym = nn.Parameter(sym_graph[self.m_sym])
        self.adap = nn.Parameter(torch.zeros_like(adj_graph))

        self.joint_embed = embed(3, 64, self.num_joints, norm=True, bias=bias)
        self.joint_dif_embed = embed(2, 64, self.num_joints, norm=True, bias=bias)
        self.bone_embed = embed(3, 64, self.num_joints, norm=True, bias=bias)
        self.bone_dif_embed = embed(2, 64, self.num_joints, norm=True, bias=bias)
        self.conv_cat = nn.Conv2d(4 * 64, 128, 1, bias=bias)
        self.bn_cat = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.limb_blocks, self.dim1, 2 * self.dim1, bias=bias)
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)

        self.fc = nn.Linear(self.dim1 * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

    def forward(self, input):
        # Dynamic Representation
        bs, step, dim = input.size()
        num_joints = dim // 3
        input = input.view((bs, step, num_joints, 3))
        input = input.permute(0, 3, 2, 1).contiguous()  # (B, C, N, T)

        joint_dif = input[:, :2, :, 1:] - input[:, :2, :, 0:-1]
        joint_dif = torch.cat([joint_dif.new(bs, joint_dif.size(1), num_joints, 1).zero_(), joint_dif], dim=-1)

        # pairs[:-2]: remove (11, 8), (5, 2) pairs
        bone = torch.zeros_like(input)
        for v1, v2 in self.pairs[:-2]:
            bone[:, :2, v1, :] = input[:, :2, v1, :] - input[:, :2, v2, :]
        bone[:, 2:, :, :] = input[:, 2:, :, :]
        bone_dif = bone[:, :2, :, 1:] - bone[:, :2, :, 0:-1]
        bone_dif = torch.cat([bone_dif.new(bs, bone_dif.size(1), num_joints, 1).zero_(), bone_dif], dim=-1)

        joint = self.joint_embed(input)
        joint_dif = self.joint_dif_embed(joint_dif)
        bone = self.bone_embed(bone)
        bone_dif = self.bone_dif_embed(bone_dif)

        # Combine
        input = torch.cat([joint_dif, joint, bone, bone_dif], 1)
        input = self.relu(self.bn_cat(self.conv_cat(input)))

        adj = torch.zeros_like(self.adj_graph)
        adj[self.m_adj] = self.e_adj
        sym = torch.zeros_like(self.sym_graph)
        sym[self.m_sym] = self.e_sym
        g = self.compute_g1(input)
        g = g + adj + sym + self.adap
        input = self.gcn1(input, g)
        input = self.gcn2(input, g)
        input = self.gcn3(input, g)

        input = self.cnn(input)
        # Classification
        output = self.maxpool(input)
        output = torch.flatten(output, 1)
        output = self.fc(output)

        return output


class norm_data(nn.Module):
    def __init__(self, dim=64, num_joints=25):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim * num_joints)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x


class embed(nn.Module):
    def __init__(self, dim=3, dim1=128, num_joints=25, norm=False, bias=False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim, num_joints),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x


class cnn1x1(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x


class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)

    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x


class compute_g_spa(nn.Module):
    def __init__(self, dim1=64 * 3, dim2=64 * 3, bias=False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):
        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g


class local(nn.Module):
    def __init__(self, limb_blocks, dim1=3, dim2=3, bias=False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.multi_spatial = multi_scale_spatial(limb_blocks)
        self.concat_conv = nn.Conv2d(2*dim1, dim2, 1, bias=bias)
        self.concat_bn = nn.BatchNorm2d(dim2)

        self.cnn1 = nn.Conv2d(dim2, dim2, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim2, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.20)

    def forward(self, x1):
        y1 = self.maxpool(x1)
        y2 = self.multi_spatial(x1)
        y = torch.cat((y1, y2), dim=1)
        x = self.dropout(self.relu(self.concat_bn(self.concat_conv(y))))

        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class multi_scale_spatial(nn.Module):
    def __init__(self, limb_blocks):
        super(multi_scale_spatial, self).__init__()
        self.left_arm, self.right_arm, self.left_leg, self.right_leg, self.head_spine = limb_blocks

        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 20))
        self.maxpool2 = nn.AdaptiveMaxPool2d((1, 20))
        self.maxpool3 = nn.AdaptiveMaxPool2d((1, 20))
        self.maxpool4 = nn.AdaptiveMaxPool2d((1, 20))
        self.maxpool5 = nn.AdaptiveMaxPool2d((1, 20))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 20))

    def forward(self, x):
        # x:(B, C, N, T)
        ll = self.maxpool1(x[:, :, self.left_leg])
        rl = self.maxpool2(x[:, :, self.right_leg])
        la = self.maxpool3(x[:, :, self.left_arm])
        ra = self.maxpool4(x[:, :, self.right_arm])
        hs = self.maxpool5(x[:, :, self.head_spine])

        multi_sptial = torch.cat((ll, rl, la, ra, hs), dim=-2)
        x = self.avgpool(multi_sptial)

        return x


if __name__ == "__main__":
    import torch
    import numpy as np
    import argparse
    from tools.skeleton import Skeleton

    parser = argparse.ArgumentParser(description='Setting training strategy')
    parser.add_argument('--train', type=int, default=1, help='The training mode')
    parser.add_argument('--batch-size', type=int, default=2, help='The size of mini batch')
    parser.add_argument('--data', type=str, default='ntu', help='The type of dataset')
    args = parser.parse_args()

    skeleton = Skeleton('NTU60')
    model = ESN(60, skeleton, 20)
    model = model.cuda()

    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()

    print('INFO: Trainable parameter count:', model_params)
    input = torch.randn(2, 20, 75)
    input = input.cuda()

    # summary(model, (27, 15, 2))
    output = model(input)
    print(output.shape)
