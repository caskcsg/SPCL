import matplotlib.pyplot as plt
import numpy as np
import torch
def gen_rand_point(x,y, r, num, label):
    ret = []
    for _ in range(num):
        _r = np.random.rand(1)[0] * r
        theta = np.random.rand(1)[0] * np.pi * 2
        _x = _r * np.cos(theta) + x
        _y = _r * np.sin(theta) + y
        ret.append((_x, _y, label))
    return ret

def dif( x, y, center, centers):
    p = np.array([x,y])
    dis_sum = 0
    for _center in centers:
        c = np.array(_center)
        dis_sum += np.linalg.norm(p-c)
    return np.linalg.norm(p-np.array(center)) / dis_sum

if __name__ == '__main__':
    np.random.seed(0)
    centers = [[0,0], [2,5], [-0.5,4]]
    rs = [2,2,2.5]
    points = []
    color = {}
    marker = {}
    color['class-0'] = 'xkcd:bright sky blue'
    color['class-1'] = 'cornflowerblue'
    color['class-2'] = 'hotpink'
    color['selection'] = 'limegreen'
    color['no-selection'] = 'slategrey'
    marker['class-0'] = 'p'
    marker['class-1'] = 'x'
    marker['class-2'] = 'D'
    marker['selection'] = '*'
    marker['no-selection'] = 'o'

    for idx in range(3):
        x,y = centers[idx]
        points.extend(gen_rand_point(x,y, rs[idx], 300 + idx * 50, idx))
    
    tensor_points = []
    tensor_labels = []
    for p in points:
        tensor_points.extend([p[0], p[1]])
        tensor_labels.append(p[2])
    tensor_points = torch.FloatTensor(tensor_points).reshape(-1, 2)
    tensor_labels = torch.FloatTensor(tensor_labels)

    scores = []
    for p in points:
        center_idx = p[2]
        center = centers[center_idx]
        scores.append(dif(p[0], p[1], center, centers))
    sorted_idxs = np.argsort(np.array(scores))
    sorted_idxs = torch.LongTensor(sorted_idxs)


    epochs = 10

    size = 3
    fig = plt.figure(figsize=(9, 3))
    fig.tight_layout()
    plt.rc('font', family='Times New Roman')
    fz = 20
    plt.subplot(1,3,1)
    for label in range(3):
        idxs = (tensor_labels == label).long()
        num_data = idxs.sum()
        idxs = torch.argsort(idxs)[-num_data:]
        class_name = 'class-{}'.format(label)
        plt.scatter(tensor_points[idxs, 0],
                    tensor_points[idxs, 1], 
                    label=class_name, c = color[class_name], marker=marker[class_name], s=size, alpha=0.7)
    plt.title('A. Original data', fontsize=fz, y=-0.2)
    # plt.legend(fontsize=fz)

    num_data = len(scores)
# --------------------------
    plt.subplot(1,3,2)
    no_selec = plt.scatter(tensor_points[:, 0],
                tensor_points[:, 1], 
                label='-', c = color['no-selection'], marker=marker[class_name], s=size)
    idxs = sorted_idxs[:int(num_data*0.2)]
    selec = plt.scatter(tensor_points[idxs, 0],
                tensor_points[idxs, 1], 
                label='selected', c = color['selection'], marker=marker['selection'], s=size)
    plt.title('B. The easiest 20%', fontsize=fz, y=-0.2)
    # plt.legend([selec], ['selected'], loc=4, fontsize=fz)
#---------------------------
    plt.subplot(1,3,3)
    no_selec = plt.scatter(tensor_points[:, 0],
                tensor_points[:, 1], 
                label='-', c = color['no-selection'], marker=marker[class_name], s=size)
    idxs = sorted_idxs[-int(num_data*0.2):]
    selec = plt.scatter(tensor_points[idxs, 0],
                tensor_points[idxs, 1], 
                label='selected', c = 'r', marker=marker['selection'], s=size)
    plt.title('C. The hardest 20%', fontsize=fz, y=-0.2)
    # plt.legend([selec], ['selected'], loc=4, fontsize=fz)
#---------------------------
    plt.savefig('./difficult.pdf',bbox_inches='tight', dpi=600)
    plt.show()