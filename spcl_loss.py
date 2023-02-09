from config import *

class SupProtoConLoss(nn.Module):
    def __init__(self, num_classes, temp, pool_size, support_set_size, centers):
        super().__init__()
        self.temperature = temp
        self.default_centers = centers.squeeze()
        self.pools = {}
        for idx in range(num_classes):
            self.pools[idx] = [self.default_centers[idx]]
        self.num_classes = num_classes
        self.pool_size = pool_size
        self.K = support_set_size
        self.eps = 1e-8
    
    def score_func(self, x, y):
        return (1+F.cosine_similarity(x, y, dim=-1))/2 + self.eps
    
    def forward(self, reps, labels, decoupled=False):
        batch_size = reps.shape[0]
        curr_centers = []
        pad_labels = []
        # calculate temporary centers
        for idx in range(self.num_classes):
            if len(self.pools[idx]) >= self.K:
            # if len(self.pools[idx]) > 0:
                tensor_center = torch.stack(self.pools[idx], 0)
                perm = torch.randperm(tensor_center.size(0))
                select_idx = perm[:self.K]
                curr_centers.append(tensor_center[select_idx].mean(0))
                pad_labels.append(idx)
            else:
                curr_centers.append(self.default_centers[idx])
                pad_labels.append(idx)
        curr_centers = torch.stack(curr_centers, 0)
        pad_labels = torch.LongTensor(pad_labels).to(reps.device)
        
        # update representations pools
        for idx in range(batch_size):
            label = labels[idx].item()
            self.pools[label].append(reps[idx].detach())
            random.shuffle(self.pools[label])
            self.pools[label] = self.pools[label][-self.pool_size:]
        
        concated_reps = torch.cat((reps, curr_centers), 0)
        concated_labels = torch.cat((labels, pad_labels), 0)
        concated_bsz = batch_size + curr_centers.shape[0]
        mask1 = concated_labels.unsqueeze(0).expand(concated_labels.shape[0], concated_labels.shape[0])
        mask2 = concated_labels.unsqueeze(1).expand(concated_labels.shape[0], concated_labels.shape[0])
        mask = 1 - torch.eye(concated_bsz).to(reps.device)
        pos_mask = (mask1 == mask2).long()
        rep1 = concated_reps.unsqueeze(0).expand(concated_bsz, concated_bsz, concated_reps.shape[-1])
        rep2 = concated_reps.unsqueeze(1).expand(concated_bsz, concated_bsz, concated_reps.shape[-1])
        scores = self.score_func(rep1, rep2)
        scores *= 1 - torch.eye(concated_bsz).to(scores.device)
        
        scores /= self.temperature
        scores = scores[:batch_size]
        pos_mask = pos_mask[:batch_size]
        mask = mask[:batch_size]
        scores -= torch.max(scores).item()
        
        if decoupled:
            pos_scores = scores * (pos_mask * mask)
            pos_scores = pos_scores.sum(-1) / ((pos_mask * mask).sum(-1) + self.eps)
            neg_scores = torch.exp(scores) * (1 - pos_mask)
            loss = -pos_scores + torch.log(neg_scores.sum(-1)+self.eps)
            loss_mask = (loss > 0).long()
            loss = (loss * loss_mask).sum() / (loss_mask.sum().item() + self.eps)
        else:
            scores = torch.exp(scores)
            pos_scores = scores * (pos_mask * mask)
            neg_scores = scores * (1 - pos_mask)
            probs = pos_scores.sum(-1)/(pos_scores.sum(-1) + neg_scores.sum(-1))
            probs /= (pos_mask * mask).sum(-1) + self.eps
            loss = - torch.log(probs + self.eps)
            loss_mask = (loss > 0.3).long()
            loss = (loss * loss_mask).sum() / (loss_mask.sum().item() + self.eps)
            # loss = loss.mean()
        return loss
