import torch
import torch.nn as nn
import torch.nn.functional as F

def find_mutual_nearest_neighbors(image_features, text_features, k):
    # 归一化特征向量
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)

    # 计算余弦相似度
    image_text_similarities = torch.matmul(image_features, text_features.t())
    text_image_similarities = torch.matmul(text_features, image_features.t())

    # 找到图片特征的 k 个文本近邻
    _, image_to_text_nearest_neighbors = torch.topk(image_text_similarities, k, dim=1)
    _, text_to_image_nearest_neighbors = torch.topk(text_image_similarities, k, dim=1)

    mutual_nearest_neighbors = torch.zeros([image_features.size(0),text_features.size(0)])
    for i in range(image_features.size(0)):
        image_k_nearest = image_to_text_nearest_neighbors[i]
        text_k_nearest = text_to_image_nearest_neighbors[image_k_nearest]
        
        # 检查是否当前图片是文本的 k 近邻之一
        has_mutual = torch.where(text_k_nearest == i, 1, 0).sum(dim=1)
        mutual_text_index = has_mutual.nonzero()
        if len(mutual_text_index) !=0:
            for idx in mutual_text_index:
                mutual_nearest_neighbors[i,image_k_nearest[idx]] = 1


    return mutual_nearest_neighbors

def compute_part(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-6):
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).to(torch.int)

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta
    # ['trousers','genders','belongings','shoes','top','hairstyle']
    # text_proj_text =  F.softmax(logit_scale * text_norm @ text_norm.t(),dim=1)
    # labels = torch.where(text_proj_text>0.4,1,0) | labels
    k = 5 if text_fetures.size(0) > 5 else text_fetures.size(0)
    t2i_labels = find_mutual_nearest_neighbors(text_fetures,image_fetures,k=k).to(text_fetures.device).to(torch.int)
    i2t_labels = find_mutual_nearest_neighbors(image_fetures,text_fetures,k=k).to(text_fetures.device).to(torch.int)
    
    labels = (i2t_labels * t2i_labels) | labels
    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def compute_patch(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size,length,num_dim = image_fetures.size() #32,192,512
    patch_feats = image_fetures.reshape(batch_size*length,num_dim)
    patch_feats = patch_feats / patch_feats.norm(dim=1, keepdim=True)
    text_fetures = text_fetures/ text_fetures.norm(dim=1, keepdim=True)
    sim_i2t = patch_feats @ text_fetures.t()
    sim_t2i = text_fetures @ patch_feats.t()

    label_i2t = sim_i2t.argmax(dim=-1)
    label_t2i = sim_t2i.argmax(dim=-1)
    loss = F.cross_entropy(logit_scale * sim_i2t, label_i2t) + F.cross_entropy(logit_scale * sim_t2i, label_t2i)
    
    return loss


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

