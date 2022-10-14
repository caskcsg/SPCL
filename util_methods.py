from config import *


def gen_all_reps(model, data):
    model.eval()
    '''
    获取当前模型下所有样本的表示以及对应标签，用这里的输出去做聚类
    '''
    results = []
    label_results = []

    sampler = SequentialSampler(data)
    dataloader = DataLoader(
        data,
        batch_size=CONFIG['batch_size'],
        sampler=sampler,
        num_workers=0  # multiprocessing.cpu_count()
    )
    inner_model = model.module if hasattr(model, 'module') else model
    tq_train = tqdm(total=len(dataloader), position=1)
    tq_train.set_description("generate representations for all data")
    with torch.no_grad():
        for batch_id, batch_data in enumerate(dataloader):
            batch_data = [x.to(inner_model.device()) for x in batch_data]
            sentences = batch_data[0]
            emotion_idxs = batch_data[1]
            
            outputs = inner_model.gen_f_reps(sentences)
            outputs = outputs.reshape(-1, outputs.shape[-1])
            for idx, label in enumerate(emotion_idxs.reshape(-1)):
                if label < 0:
                    continue
                results.append(outputs[idx])
                label_results.append(label)
            tq_train.update()
    tq_train.close()
    dim = results[0].shape[-1]

    results = torch.stack(results, 0).reshape(-1, dim)
    label_results = torch.stack(label_results, 0).reshape(-1)

    return results, label_results


def cluster(reps, labels, init_centers=None, init_centers_mask=None, epoch=0):

    label_space = {}
    label_space_dataid = {}
    centers = []
    for idx in range(CONFIG['num_classes']):
        label_space[idx] = []
        label_space_dataid[idx] = []
    for idx, turn_reps in enumerate(reps):
        emotion_label = labels[idx].item()
        if emotion_label < 0:
            continue
        label_space[emotion_label].append(turn_reps)
        label_space_dataid[emotion_label].append(idx)
    # clustering for each emotion class
    dim = label_space[0][0].shape[-1]

    max_num_clusters = 0
    cluster2dataid = {}
    cluster2classid = {}
    total_clusters = 0
    all_centers = []
    for emotion_label in range(CONFIG['num_classes']):

        x = torch.stack(label_space[emotion_label], 0).reshape(-1, dim)
        # if init_centers is not None and init_centers_mask is not None:
        #     init = init_centers[
        #         emotion_label, :init_centers_mask[emotion_label].sum(), :]
        # else:
        #     init = []
        # kmeans_pytorch
        # num_clusters = x.shape[0] // CONFIG['avg_cluster_size']
        # if num_clusters > 1:
        #     flag = True
        #     while flag and num_clusters > 1:
        #         flag = False
        #         cluster_idxs, cluster_centers = kmeans(
        #             X=x,
        #             num_clusters=num_clusters,
        #             cluster_centers=[],
        #             distance=CONFIG['dist_func'],
        #             device=torch.device('cpu'),
        #             tqdm_flag=False,
        #         )
        #         for c_idx in range(num_clusters):
        #             c_size = (cluster_idxs == c_idx).sum()
        #             if c_size < CONFIG['avg_cluster_size']//2:
        #                 flag = True
        #                 num_clusters -= 1
        #             logging.info('decrease num_cluster')
        # if num_clusters <= 1:
        num_clusters = 1
        cluster_idxs = torch.zeros(x.shape[0]).long()
        cluster_centers = x.mean(0).unsqueeze(0).cpu()
        logging.info('{} clusters for emotion {}'.format(num_clusters, emotion_label))
        centers.append(cluster_centers)

        max_num_clusters = max(num_clusters, max_num_clusters)
        # 记录聚类中心到数据索引的映射，由此来构造对比学习的样本
        cluster_idxs += total_clusters
        for d_idx, c_idx in enumerate(cluster_idxs.numpy().tolist()):
            if c_idx < 0:
                continue
            if cluster2dataid.get(c_idx) is None:
                cluster2dataid[c_idx] = []
            cluster2classid[c_idx] = emotion_label
            cluster2dataid[c_idx].append(
                label_space_dataid[emotion_label][d_idx])
        total_clusters += num_clusters
        for c_idx in range(num_clusters):
            all_centers.append(cluster_centers[c_idx, :])
    
    centers_mask = []
    for emotion_label in range(CONFIG['num_classes']):
        num_clusters, dim = centers[emotion_label].shape[0], centers[
            emotion_label].shape[-1]
        centers_mask.append(torch.zeros(max_num_clusters))
        centers_mask[emotion_label][:num_clusters] = 1
        centers[emotion_label] = torch.cat(
            (centers[emotion_label],
             torch.ones(max_num_clusters - num_clusters, dim)), 0)
    centers = torch.stack(centers, 0).to(CONFIG['device'])
    centers_mask = torch.stack(centers_mask, 0).to(CONFIG['device'])
    return centers, centers_mask, cluster2dataid, cluster2classid, all_centers


def plot_data(reps, labels, epoch, selection=None):

    emotion_vocab = vocab.Vocab.from_dict(torch.load(CONFIG['emotion_vocab']))
    plt.figure(figsize=(32, 16))
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
    tsne_res = tsne.fit_transform(reps.cpu().numpy())
    for emotion_label in range(CONFIG['num_classes']):
        idxs = (labels == emotion_label).long().cpu()
        num_data = idxs.sum()
        idxs = torch.argsort(idxs)[-num_data:]
        plt.subplot(1,2,1)
        plt.scatter(tsne_res[idxs, 0],
                    tsne_res[idxs, 1],
                    label=emotion_vocab.index2word(emotion_label), s=100)
        plt.subplot(1,2,2)
        plt.scatter(tsne_res[idxs, 0],
                    tsne_res[idxs, 1],
                    label=emotion_vocab.index2word(emotion_label), s=100)

    if selection is not None:
        plt.subplot(1,2,2)
        plt.scatter(
            tsne_res[selection, 0],
            tsne_res[selection, 1],
            label='selection',
            s=100
        )
    plt.legend()
    plt.savefig(CONFIG['temp_path'] + '/cluster_results/{}_cluster.jpg'.format(epoch))
    plt.close()

def get_kth(rows, ratio=CONFIG['ratio']):
    num_data = rows.shape[-1]
    num_used = (rows==0).sum().item()
    if num_data == num_used:
        return -1, -1
    kth = max(1, int((num_data - num_used) * ratio))
    return torch.kthvalue(rows, kth)
    

def selection(reps, all_centers, cluster2dataid, selection_ratio):
    
    total_cluster = len(all_centers)
    data2clusterid = {}
    for c_idx in range(total_cluster):
        for data_id in cluster2dataid[c_idx]:
            data2clusterid[data_id] = c_idx
    all_centers = torch.stack(all_centers, 0).to(reps.device)
    # difficult measure function
    dis_scores = []
    for idx, rep in enumerate(reps):
        self_center = all_centers[data2clusterid[idx]]
        self_dis = dist(rep, self_center)
        sum_dis = dist(
            rep.unsqueeze(0).expand_as(all_centers),
            all_centers
            )
        dis_scores.append(self_dis/sum_dis.sum())
    dis_scores = torch.FloatTensor(dis_scores)
    priority_seq = torch.argsort(dis_scores, descending=False).cpu().numpy().tolist()

    num_selection = int(selection_ratio * len(priority_seq))
    select_data_idx = priority_seq[:num_selection]
    
    return select_data_idx
def gen_cl_data(reps,
                all_centers,
                cluster2dataid,
                cluster2classid,
                epoch=0):

    batch_size = CONFIG['batch_size']
    num_data = reps.shape[0]
    dim = reps.shape[-1]
    total_cluster = len(all_centers)

    cluster_idxs = torch.zeros(num_data).long()
    labels = torch.zeros(num_data).long()

    for c_idx in range(total_cluster):
        for data_id in cluster2dataid[c_idx]:
            cluster_idxs[data_id] = c_idx
            labels[data_id] = cluster2classid[c_idx]
    seed_list = selection(reps, all_centers, cluster2dataid, CONFIG['ratio'])
    # plot_data(reps, labels, epoch, seed_list)
    return seed_list, cluster_idxs