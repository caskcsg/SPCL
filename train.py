import transformers
from config import *
from model import CLModel, FGM

from data_process import *
from util_methods import *
from spcl_loss import SupProtoConLoss

def get_paramsgroup(model, warmup=False):
    no_decay = ['bias', 'LayerNorm.weight']
    pre_train_lr = CONFIG['ptmlr']

    bert_params = list(map(id, model.f_context_encoder.parameters()))
    params = []
    warmup_params = []
    for name, param in model.named_parameters():
        lr = CONFIG['lr']
        weight_decay = 0.01
        if id(param) in bert_params:
            lr = pre_train_lr
        if any(nd in name for nd in no_decay):
            weight_decay = 0
        params.append({
            'params': param,
            'lr': lr,
            'weight_decay': weight_decay
        })
        warmup_params.append({
            'params':
            param,
            'lr':
            CONFIG['ptmlr'] / 4 if id(param) in bert_params else lr,
            'weight_decay':
            weight_decay
        })
    if warmup:
        return warmup_params
    params = sorted(params, key=lambda x: x['lr'])
    return params


def train_epoch(model,
                optimizer,
                lr_scheduler,
                trainset,
                centers,
                centers_mask=None,
                epoch=0,
                max_step=-1,
                train_obj='all'):
    model.train()
    inner_model = model.module if hasattr(model, 'module') else model
    ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    spcl_loss = SupProtoConLoss(
        num_classes=CONFIG['num_classes'],
        temp=CONFIG['temperature'],
        pool_size=CONFIG['pool_size'],
        support_set_size=CONFIG['support_set_size'],
        centers=centers)
    accumulation_steps = CONFIG['accumulation_steps']
    sampler = RandomSampler(
        trainset) if CONFIG['local_rank'] == -1 else DistributedSampler(
            trainset)
    if CONFIG['local_rank'] != -1:
        sampler.set_epoch(epoch)
    dataloader = DataLoader(trainset,
                            batch_size=CONFIG['batch_size'],
                            sampler=sampler,
                            num_workers=0,
                            drop_last=True)
    
    tq_train = tqdm(total=max_step if max_step > 0 else len(dataloader),
                    position=1,
                    disable=CONFIG['local_rank'] not in [-1, 0])
    def calc_loss(sentences, emotion_idxs, labels):
        ccl_reps = inner_model.gen_f_reps(sentences)
        
        if train_obj == 'ce':
            direct_loss = ce_loss_func(inner_model.predictor(ccl_reps),
                                    emotion_idxs)
        else:
            direct_loss = ce_loss_func(inner_model.predictor(ccl_reps.detach()),
                                   emotion_idxs)
        ins_cl_loss = torch.zeros(1).to(CONFIG['device'])
        if train_obj == 'spcl':
            ins_cl_loss = spcl_loss(ccl_reps, labels)
        if train_obj == 'spdcl':
            ins_cl_loss = spcl_loss(ccl_reps, labels, decoupled=True)

        loss = direct_loss + ins_cl_loss

        tq_train.set_description(
            'direct loss {:.2f}, instance_cl_loss {:.2f}'
            .format(direct_loss.item(),
                    ins_cl_loss.item() if train_obj in ['spcl', 'spdcl'] else 0))
        return loss
        
    fgm = FGM(inner_model)
    for batch_id, batch_data in enumerate(dataloader):
        batch_data = [x.to(inner_model.device()) for x in batch_data]

        sentences = batch_data[0]

        emotion_idxs = batch_data[1].reshape(-1)
        labels = batch_data[2].reshape(-1)

        loss = calc_loss(sentences, emotion_idxs, labels)
        loss = loss / accumulation_steps
        loss.backward()
        if CONFIG['fgm']:
            fgm.attack()
            loss = calc_loss(sentences, emotion_idxs, labels)
            loss = loss / accumulation_steps
            loss.backward()
            fgm.restore()
        
        nn.utils.clip_grad_norm_(inner_model.parameters(), max_norm=5, norm_type=2)
        tq_train.update()

        if batch_id % accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if max_step > 0 and batch_id > max_step:
            optimizer.zero_grad()
            break
    optimizer.zero_grad()
    tq_train.close()

def train(model, train_dialogues, dev_dialogues, test_dialogues):

    devset = build_dataset(dev_dialogues)
    testset = build_dataset(test_dialogues)

    tq_epoch = tqdm(total=CONFIG['epochs'],
                    position=0,
                    disable=CONFIG['local_rank'] not in [-1, 0])
    centers = None

    if CONFIG['local_rank'] != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[CONFIG['local_rank']],
            output_device=CONFIG['local_rank'],
            find_unused_parameters=True)
    optimizer = torch.optim.AdamW(
        get_paramsgroup(
            model.module if hasattr(model, 'module') else model))
    # train
    best_f1 = -1
    epochs_not_update = 0
    train_obj = CONFIG['train_obj']
    for epoch in range(CONFIG['epochs']+1):
        tq_epoch.set_description('training on epoch {}'.format(epoch))
        tq_epoch.update()

        trainset = build_dataset(train_dialogues, train=True)

        if CONFIG['local_rank'] in [-1, 0]:
            all_reps, all_corr_labels = gen_all_reps(model, trainset)
            
            logging.info('clustering...')
            centers, centers_mask, cluster2dataid, cluster2classid, all_centers = cluster(
                all_reps, all_corr_labels, init_centers=centers, epoch=epoch)
            num_data = all_reps.shape[0]
            if epoch > 0:
                f1 = test(model,
                          devset,
                          centers,
                          centers_mask,
                          desc='dev @ epoch {}'.format(epoch - 1))
                if f1 > best_f1:
                    epochs_not_update = 0
                    os.system('rm {}/models/*'.format(CONFIG['temp_path']))
                    os.system('rm {}/centers/*'.format(CONFIG['temp_path']))

                    best_f1 = f1
                    model_to_save = model.module if hasattr(
                        model, "module") else model
                    torch.save(
                        model_to_save, CONFIG['temp_path'] +
                        '/models/f1_{:.4f}_@epoch{}.pkl'.format(
                            best_f1, epoch - 1))
                    torch.save(
                        centers, CONFIG['temp_path'] +
                        '/centers/f1_{:.4f}_@epoch{}.pkl'.format(
                            best_f1, epoch - 1))
                    torch.save(
                        centers_mask, CONFIG['temp_path'] +
                        '/centers/f1_{:.4f}_@epoch{}.msk'.format(
                            best_f1, epoch - 1))

                    f1 = test(model,
                              testset,
                              centers,
                              centers_mask,
                              desc='new best test @ epoch {}'.format(
                                  epoch - 1))
                else:
                    epochs_not_update += 1
                if epochs_not_update >= 3:
                    break

            selection, cluster_idxs = gen_cl_data(all_reps,
                                                  all_centers,
                                                  cluster2dataid,
                                                  cluster2classid,
                                                  epoch=epoch)
            # st = 1 - epoch / 10
            # ed = epoch / 10
            st = 1 - epoch / CONFIG['epochs']
            ed = epoch / CONFIG['epochs']
            num_data = len(selection)
            selection = torch.LongTensor(selection)
            prob_list = [
                st + (ed - st) / (num_data - 1) * i for i in range(num_data)
            ]
            prob_tensor = torch.FloatTensor(prob_list)
            rand_prob_tensor = torch.bernoulli(torch.ones(num_data) * 0.5)
            if CONFIG['cl']:
                sample = torch.bernoulli(prob_tensor).long()
            else:
                sample = torch.bernoulli(rand_prob_tensor).long()
            selection = selection * sample
            sample_results = selection[torch.nonzero(selection)]
            all_idxs = sample_results.numpy().tolist()

            epoch_trainset = TensorDataset(trainset[all_idxs][0],
                                           trainset[all_idxs][1],
                                           cluster_idxs[all_idxs])
            torch.save(epoch_trainset,
                       CONFIG['temp_path'] + '/temp/train_set.pkl')
            torch.save(centers, CONFIG['temp_path'] + '/temp/centers.pkl')
            torch.save(centers_mask,
                       CONFIG['temp_path'] + '/temp/centers_mask.pkl')
        if CONFIG['local_rank'] != -1:
            torch.distributed.barrier()

        epoch_trainset = torch.load(CONFIG['temp_path'] +
                                    '/temp/train_set.pkl',
                                    map_location=CONFIG['device'])
        centers = torch.load(CONFIG['temp_path'] + '/temp/centers.pkl',
                             map_location=CONFIG['device'])
        centers_mask = torch.load(CONFIG['temp_path'] +
                                  '/temp/centers_mask.pkl',
                                  map_location=CONFIG['device'])
        num_training_steps = len(epoch_trainset)//(CONFIG['batch_size'] * CONFIG['accumulation_steps'])
        num_warmup_steps = min(CONFIG['warm_up'], num_training_steps) if epoch == 0 else 0
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                                    optimizer,
                                    num_warmup_steps=num_warmup_steps,
                                    num_training_steps=num_training_steps
                                    )
        if epoch < CONFIG['epochs']:
            train_epoch(model,
                    optimizer,
                    lr_scheduler,
                    epoch_trainset,
                    centers,
                    centers_mask,
                    epoch,
                    train_obj=train_obj)

    tq_epoch.close()
    if CONFIG['local_rank'] in [0, -1]:
        model, centers, centers_mask = load_latest()
        f1 = test(model, testset, centers, centers_mask)
        print('best f1 test is {:.4f}'.format(f1), flush=True)
        os.system('rm -r {}'.format(CONFIG['temp_path']))


def test(model, data, centers, centers_mask, desc=''):

    model.eval()
    inner_model = model.module if hasattr(model, 'module') else model
    y_true_list = []
    direct_list = [[], [], [], []]
    cluster_list = [[], [], [], []]
    sampler = SequentialSampler(data)
    dataloader = DataLoader(
        data,
        batch_size=CONFIG['batch_size'],
        sampler=sampler,
        num_workers=0,  # multiprocessing.cpu_count()
    )
    tq_test = tqdm(total=len(dataloader), desc="testing", position=2)
    for batch_id, batch_data in enumerate(dataloader):
        batch_data = [x.to(inner_model.device()) for x in batch_data]
        sentences = batch_data[0]
        emotion_idxs = batch_data[1].reshape(-1)
        with torch.no_grad():
            ccl_reps = inner_model.gen_f_reps(sentences)
        cluster_outputs, direct_outputs = [], []

        feature_list = [ccl_reps]
        num_feature = len(feature_list)

        for idx, feature in enumerate(feature_list):
            #
            outputs = inner_model(feature, centers, score_func)
            outputs -= (1 - centers_mask) * 2
            cluster_outputs.append(torch.argmax(outputs.max(-1)[0], -1))
            direct_outputs.append(
                torch.argmax(inner_model.predictor(feature), -1))
        for batch_id in range(emotion_idxs.shape[0]):
            if emotion_idxs[batch_id] > -1:
                for idx in range(num_feature):
                    direct_list[idx].append(
                        direct_outputs[idx][batch_id].item())
                    cluster_list[idx].append(
                        cluster_outputs[idx][batch_id].item())
                y_true_list.append(emotion_idxs[batch_id].item())
        tq_test.update()
    direct_f1_scores = [
        f1_score(y_true=y_true_list,
                 y_pred=direct_list[idx],
                 average='weighted') for idx in range(num_feature)
    ]
    cluster_f1_scores = [
        f1_score(y_true=y_true_list,
                 y_pred=cluster_list[idx],
                 average='weighted') for idx in range(num_feature)
    ]

    # f1 = max(max(direct_f1_scores), max(cluster_f1_scores))
    f1 = max(cluster_f1_scores)
    print('\n{} best w-f1 is {:.4f}'.format(desc, f1), flush=True)
    print('direct f1 cls {:.4f}'.format(*direct_f1_scores))
    print('cluster f1 cls {:.4f}'.format(*cluster_f1_scores),
          flush=True)

    return f1


def load_latest():
    model_path = CONFIG['temp_path'] + '/models'
    lst = os.listdir(model_path)
    lst = list(filter(lambda item: item.endswith('.pkl'), lst))
    lst.sort(key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
    model = torch.load(os.path.join(model_path, lst[-1]))
    logging.info(
        'model checkpoint {} is loaded'.format(
            os.path.join(model_path, lst[-1])), )
    center_path = CONFIG['temp_path'] + '/centers'
    lst = os.listdir(center_path)
    lst = list(filter(lambda item: item.endswith('.pkl'), lst))
    lst.sort(key=lambda x: os.path.getmtime(os.path.join(center_path, x)))
    centers = torch.load(os.path.join(center_path, lst[-1]))
    logging.info(
        'center checkpoint {} is loaded'.format(
            os.path.join(center_path, lst[-1])), )

    lst = os.listdir(center_path)
    lst = list(filter(lambda item: item.endswith('.msk'), lst))
    lst.sort(key=lambda x: os.path.getmtime(os.path.join(center_path, x)))
    centers_mask = torch.load(os.path.join(center_path, lst[-1]))
    logging.info(
        'centers mask checkpoint {} is loaded'.format(
            os.path.join(center_path, lst[-1])), )

    return model, centers, centers_mask


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-te',
                        '--test',
                        action='store_true',
                        help='run test',
                        default=False)
    parser.add_argument('-tr',
                        '--train',
                        action='store_true',
                        help='run train',
                        default=False)
    parser.add_argument('-ft',
                        '--finetune',
                        action='store_true',
                        help='fine tune the best model',
                        default=False)
    parser.add_argument('-cl',
                        '--cl',
                        action='store_true',
                        help='use CL',
                        default=False)
    parser.add_argument('-pr',
                        '--print_error',
                        action='store_true',
                        help='print error case',
                        default=False)
    parser.add_argument('-mlp',
                        '--output_mlp',
                        action='store_true',
                        help='use an additional mlp layer on the model output',
                        default=False)
    parser.add_argument('-fgm',
                        '--fgm',
                        action='store_true',
                        help='use fgm',
                        default=False)
    parser.add_argument('-bsz',
                        '--batch_size',
                        help='Batch_size per gpu',
                        required=False,
                        default=CONFIG['batch_size'],
                        type=int)
    parser.add_argument('-seed',
                        '--seed',
                        help='seed',
                        required=False,
                        default=42,
                        type=int)
    parser.add_argument('-psz',
                        '--pool_size',
                        help='Batch_size per gpu',
                        required=False,
                        default=CONFIG['pool_size'],
                        type=int)
    parser.add_argument('-ssz',
                        '--support_set_size',
                        help='support size per gpu',
                        required=False,
                        default=CONFIG['support_set_size'],
                        type=int)
    parser.add_argument('-epochs',
                        '--epochs',
                        help='epochs',
                        required=False,
                        default=CONFIG['epochs'],
                        type=int)
    parser.add_argument('-cluster_size',
                        '--avg_cluster_size',
                        help='avg_cluster_size',
                        required=False,
                        default=CONFIG['avg_cluster_size'],
                        type=int)
    parser.add_argument('-lr',
                        '--lr',
                        help='learning rate',
                        required=False,
                        default=CONFIG['lr'],
                        type=float)
    parser.add_argument('-ptmlr',
                        '--ptmlr',
                        help='ptm learning rate',
                        required=False,
                        default=CONFIG['ptmlr'],
                        type=float)
    parser.add_argument('-tsk', '--task_name', default='meld', type=str)
    parser.add_argument('-fp16',
                        '--fp_16',
                        action='store_true',
                        help='use fp 16',
                        default=False)
    parser.add_argument('-wp',
                        '--warm_up',
                        default=CONFIG['warm_up'],
                        type=int,
                        required=False)
    parser.add_argument('-dpt',
                        '--dropout',
                        default=CONFIG['dropout'],
                        type=float,
                        required=False)
    parser.add_argument('-temp',
                        '--temperature',
                        default=CONFIG['temperature'],
                        type=float,
                        required=False)
    parser.add_argument('-bert_path',
                        '--bert_path',
                        default=CONFIG['bert_path'],
                        type=str,
                        required=False)
    parser.add_argument('-train_obj',
                        '--train_obj',
                        default=CONFIG['train_obj'],
                        type=str,
                        required=False)
    parser.add_argument('-data_path',
                        '--data_path',
                        default=CONFIG['data_path'],
                        type=str,
                        required=False)
    parser.add_argument('-temp_path',
                        '--temp_path',
                        default=CONFIG['temp_path'],
                        type=str,
                        required=False)
    parser.add_argument('-acc_step',
                        '--accumulation_steps',
                        default=CONFIG['accumulation_steps'],
                        type=int,
                        required=False)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--n_gpu", type=int, default=0, help="gpu per process")
    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Whether not to use CUDA when available")

    parser.add_argument('--device',
                        default='cuda:0',
                        help='Device used for inference')
    args = parser.parse_args()
    
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device
    torch.cuda.set_device(args.local_rank)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )
    
    set_seed(args.seed)
    args_dict = vars(args)
    for k, v in args_dict.items():
        CONFIG[k] = v
    
    if CONFIG['temp_path'] == '':
        if args.local_rank in [-1]:
            os.makedirs('/test/diyi/temp', exist_ok=True)
            temp_path = tempfile.mkdtemp(dir='/test/diyi/temp')
        else:
            temp_path = '/test/diyi/temp'
    CONFIG['temp_path'] = temp_path
    CONFIG['emotion_vocab'] = CONFIG['temp_path'] + '/vocabs/emotion_vocab.pkl'

    if args.local_rank in [-1, 0]:
        os.makedirs(CONFIG['temp_path'], exist_ok=True)
        os.makedirs(os.path.join(CONFIG['temp_path'], 'vocabs'), exist_ok=True)
        os.makedirs(os.path.join(CONFIG['temp_path'], 'models'), exist_ok=True)
        os.makedirs(os.path.join(CONFIG['temp_path'], 'temp'), exist_ok=True)
        os.makedirs(os.path.join(CONFIG['temp_path'], 'centers'),
                    exist_ok=True)
        os.makedirs(os.path.join(CONFIG['temp_path'], 'cluster_results'),
                    exist_ok=True)


    if args.task_name == 'meld':
        CONFIG['data_path'] = './MELD'
        CONFIG['num_classes'] = 7
        train_data_path = os.path.join(CONFIG['data_path'], 'train_sent_emo.csv')
        test_data_path = os.path.join(CONFIG['data_path'], 'test_sent_emo.csv')
        dev_data_path = os.path.join(CONFIG['data_path'], 'dev_sent_emo.csv')
        get_meld_vocabs([train_data_path, dev_data_path, test_data_path])
        dev_dialogues = load_meld_turn(dev_data_path)
        test_dialogues = load_meld_turn(test_data_path)
        train_dialogues = load_meld_turn(train_data_path)
        
        
    if args.task_name == 'emorynlp':
        CONFIG['data_path'] = './emorynlp'
        CONFIG['num_classes'] = 7
        train_data_path = os.path.join(CONFIG['data_path'],
                                       'emotion-detection-trn.json')
        test_data_path = os.path.join(CONFIG['data_path'],
                                      'emotion-detection-tst.json')
        dev_data_path = os.path.join(CONFIG['data_path'], 'emotion-detection-dev.json')
        get_emorynlp_vocabs([train_data_path, dev_data_path, test_data_path])
        dev_dialogues = load_emorynlp_turn(dev_data_path)
        test_dialogues = load_emorynlp_turn(test_data_path)
        train_dialogues = load_emorynlp_turn(train_data_path)
        
        
    if args.task_name =='iemocap':
        CONFIG['data_path'] = './IEMOCAP'
        CONFIG['num_classes'] = 6
        train_data_path = os.path.join(CONFIG['data_path'], 'train_data.json')
        test_data_path = os.path.join(CONFIG['data_path'], 'test_data.json')
        dev_data_path = os.path.join(CONFIG['data_path'], 'dev_data.json')
        get_iemocap_vocabs([train_data_path, dev_data_path, test_data_path])
        dev_dialogues = load_iemocap_turn(dev_data_path)
        test_dialogues = load_iemocap_turn(test_data_path)
        train_dialogues = load_iemocap_turn(train_data_path)
        
    

    if CONFIG['local_rank'] != -1:
        torch.distributed.barrier()

    model = CLModel(CONFIG)

    if CONFIG['local_rank'] != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(args.device)
    if args.local_rank in [-1, 0]:
        print('---config---')
        for k, v in CONFIG.items():
            print(k, '\t\t\t', v, flush=True)

    if args.finetune:
        model, centers, centers_mask = load_latest()
    if args.train:
        train(model, train_dialogues, dev_dialogues, test_dialogues)
    if args.test:
        if args.task_name == 'emorynlp':
            testset = load_emorynlp_turn(test_data_path)
        if args.task_name == 'meld':
            testset = load_meld_turn(test_data_path)
        best_f1 = test(model, testset, centers, centers_mask)
        print(best_f1)
