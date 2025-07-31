import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F


class CLIPLoss(nn.Module):
    def __init__(
            self
    ):
        super().__init__()
    def forward(self, outputs, options, logit_scale):
        # get embedding data
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        if (options.distributed):
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]


            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)


            image_embeds = torch.cat(
                gathered_image_embeds[:options.the_rank] + [image_embeds] + gathered_image_embeds[options.the_rank + 1:])
            text_embeds = torch.cat(
                gathered_text_embeds[:options.the_rank] + [text_embeds] + gathered_text_embeds[options.the_rank + 1:])

        logits_text_per_image  = logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_image_per_text = logits_text_per_image.t()
        batch_size = len(logits_text_per_image)
        target = torch.arange(batch_size).long().to(options.device, non_blocking=True)
        criterion = nn.CrossEntropyLoss().to(options.device)
        clip_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
        return clip_loss

class Invert_Loss(nn.Module):
    def __init__(
            self
    ):
        super().__init__()

    def infoNCE_preds(self, similarity_matrix, temp, options):
        labels = torch.eye(len(similarity_matrix),device = options.device).float()
        # select and combine multiple positives
        pos = similarity_matrix[labels.bool()].view(similarity_matrix.shape[0], -1)
        pos = torch.exp(pos / temp).squeeze(1)
        # select only the negatives
        neg = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        neg = torch.exp(neg / temp)
        neg_sum = neg.sum(1)
        preds = pos / (pos + neg_sum)
        return preds


    def sigmoid_loss(self, sims_img_per_text, sims_img_per_no_text, options):
        N = sims_img_per_text.shape[0]
        labels_yes = 2 * torch.eye(N, device = options.device) - torch.ones(N, device = options.device)
        labels_no = -torch.eye(N, device = options.device) + (1 - torch.eye(N, device = options.device))
        loss_sig_y = -F.logsigmoid(labels_yes * (sims_img_per_text)).sum() / N
        loss_sig_n = -F.logsigmoid(labels_no * (sims_img_per_no_text)).sum() / N
        return loss_sig_y, loss_sig_n


    def relation_opposite_loss(self, S, S_no):
        cos_sim = torch.sum(S * S_no, dim=1)
        upper_loss = (cos_sim + 0.2).clamp(min=0)
        lower_loss = (-0.7 - cos_sim).clamp(min=0)
        loss = torch.mean(upper_loss + lower_loss)
        return loss


    def forward(self, outputs, options, logit_scale):
        # get embedding data
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        text_no_embeds = outputs.text_no_embeds
        if (options.distributed):
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
            gathered_text_no_embeds = [torch.zeros_like(text_no_embeds) for _ in range(options.num_devices)]

            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)
            dist.all_gather(gathered_text_no_embeds, text_no_embeds)

            image_embeds = torch.cat(
                gathered_image_embeds[:options.the_rank] + [image_embeds] + gathered_image_embeds[options.the_rank + 1:])
            text_embeds = torch.cat(
                gathered_text_embeds[:options.the_rank] + [text_embeds] + gathered_text_embeds[options.the_rank + 1:])
            text_no_embeds = torch.cat(
                gathered_text_no_embeds[:options.the_rank] + [text_no_embeds] + gathered_text_no_embeds[options.the_rank + 1:])


        # preds NC
        with torch.no_grad():
            sims_per_img = image_embeds @ text_embeds.t()
            preds = self.infoNCE_preds(sims_per_img, 1 / logit_scale.exp(), options)
            NC_num = int(options.NC_ratio * preds.size(0))
            _, batched_NC_ids = torch.topk(preds, NC_num, largest=False)
            all_ids = torch.arange(len(preds),device = options.device)
            mask = torch.ones(len(preds), dtype=torch.bool)
            mask[batched_NC_ids] = False
            batched_CC_ids = all_ids[mask]

        # mask the NC samples
        image_embeds_copy = image_embeds.clone()
        text_no_embeds_copy = text_no_embeds.clone()
        image_embeds = torch.index_select(image_embeds, 0, batched_CC_ids)
        text_embeds = torch.index_select(text_embeds, 0, batched_CC_ids)
        text_no_embeds = torch.index_select(text_no_embeds, 0, batched_CC_ids)


        #cal simlarity and logits
        batch_size = len(image_embeds)
        sims_img_per_text = image_embeds @ text_embeds.t()
        sims_img_per_no_text = image_embeds @ text_no_embeds.t()
        sims_text_per_text = text_embeds @ text_embeds.t()
        sims_no_text_per_no_text = text_no_embeds @ text_no_embeds.t()
        logits_text_per_text = logit_scale.exp() * sims_text_per_text
        logits_no_text_per_no_text = logit_scale.exp() * sims_no_text_per_no_text
        logits_img_per_text = logit_scale.exp() * sims_img_per_text
        logits_img_per_no_text = logit_scale.exp() * sims_img_per_no_text

        with torch.no_grad():
            sims_img_per_no_text = logits_img_per_no_text / logit_scale.exp()
            diagonal_mean = torch.mean(torch.diagonal(sims_img_per_no_text))
            mask = ~torch.eye(sims_img_per_no_text.shape[0], dtype=bool, device = options.device)
            off_diagonal = sims_img_per_no_text * mask
            off_diagonal_mean = torch.mean(off_diagonal)
            sims_paired_img_no_copy = torch.sum(image_embeds_copy * text_no_embeds_copy, dim=1).unsqueeze(1)
            no_NC_sims = torch.mean(sims_paired_img_no_copy[batched_NC_ids])
        inmodal_cyclic_loss = (logits_text_per_text - logits_no_text_per_no_text).square().mean() / (
                logit_scale.exp() * logit_scale.exp()) * batch_size
        cyclic_loss =  10 * ( inmodal_cyclic_loss )
        inmodal_opposite_loss = self.relation_opposite_loss(text_embeds, text_no_embeds) * 10
        cross_sig_y, cross_sig_n = self.sigmoid_loss(logits_img_per_text, logits_img_per_no_text, options)

        return batched_NC_ids, batched_CC_ids, cyclic_loss, inmodal_opposite_loss, cross_sig_y, cross_sig_n



class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def relation_opposite_loss(self, S, S_no):
        cos_sim = torch.sum(S * S_no, dim=1)
        upper_loss = (cos_sim + 0.2).clamp(min=0)
        # 低于下界的部分产生 loss
        lower_loss = (-0.7 - cos_sim).clamp(min=0)
        loss = torch.mean(upper_loss + lower_loss)
        return loss


    def sinkhorn_log_domain_torch(self, p, q, C, Mask=None, reg= 0.03, niter=100, thresh=1e-4):

        def M(u, v):
            "Modified cost for logarithmic updates"
            M = (-C + torch.unsqueeze(u, 1) + torch.unsqueeze(v, 0)) / reg
            if Mask is not None:
                M[Mask == 0] = -1e6


            return M

        def lse(A):
            "log-sum-exp"
            max_A, _ = torch.max(A, dim=1, keepdims=True)
            return torch.log(torch.exp(A - max_A).sum(1, keepdims=True) + 1e-10) + max_A  # add 10^-10to prevent NaN

        # Actual Sinkhorn loop ......................................................................
        u, v, err = 0. * p, 0. * q, 0.
        actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

        for i in range(niter):
            u1 = u  # useful to check the update
            u = reg * (torch.log(p) - lse(M(u, v)).squeeze()) + u
            v = reg * (torch.log(q) - lse(M(u, v).T).squeeze()) + v

            err = torch.sum(torch.abs(u - u1))

            actual_nits += 1
            if err < thresh:
                break
        U, V = u, v
        pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)

        # print("iter:",actual_nits)
        return pi

    def infoNCE_preds(self, similarity_matrix, temp, options):
        labels = torch.eye(len(similarity_matrix),device = options.device).float()
        # select and combine multiple positives
        pos = similarity_matrix[labels.bool()].view(similarity_matrix.shape[0], -1)
        pos = torch.exp(pos / temp).squeeze(1)
        # select only the negatives
        neg = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        neg = torch.exp(neg / temp)
        neg_sum = neg.sum(1)
        preds = pos / (pos + neg_sum)
        return preds

    def forward(self, outputs, options, logit_scale):
        # get embedding data
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        text_no_embeds = outputs.text_no_embeds
        if (options.distributed):
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
            gathered_text_no_embeds = [torch.zeros_like(text_no_embeds) for _ in range(options.num_devices)]

            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)
            dist.all_gather(gathered_text_no_embeds, text_no_embeds)

            image_embeds = torch.cat(
                gathered_image_embeds[:options.the_rank] + [image_embeds] + gathered_image_embeds[options.the_rank + 1:])
            text_embeds = torch.cat(
                gathered_text_embeds[:options.the_rank] + [text_embeds] + gathered_text_embeds[options.the_rank + 1:])
            text_no_embeds = torch.cat(
                gathered_text_no_embeds[:options.the_rank] + [text_no_embeds] + gathered_text_no_embeds[options.the_rank + 1:])
        # cal simlarity
        sims_per_img = image_embeds @ text_embeds.t()
        sims_paired_img_no = torch.sum(image_embeds * text_no_embeds, dim=1).unsqueeze(1)
        # temp scale
        logits_per_img = logit_scale.exp() * sims_per_img
        logits_paired_img_no = logit_scale.exp() * sims_paired_img_no

        # preds NC
        with torch.no_grad():
            preds = self.infoNCE_preds(sims_per_img, 1 / logit_scale.exp(), options)
            NC_num = int(options.NC_ratio * preds.size(0))
            _, batched_NC_ids = torch.topk(preds, NC_num, largest=False)

        # get Label
        with torch.no_grad():
            gamma = options.gamma
            sims_matrix = torch.cat([sims_per_img, sims_paired_img_no], dim=1)
            cost_v = 1 - sims_matrix
            cost_v = cost_v / cost_v.max()
            cost_t = 1 - sims_matrix.t()
            cost_t = cost_t / cost_t.max()

            p = torch.ones(sims_matrix.shape[0], device = options.device) / sims_matrix.shape[0]
            q = torch.ones(sims_matrix.shape[1], device = options.device) / sims_matrix.shape[1]

            M_v = torch.ones_like(cost_v, device=options.device, dtype=torch.int64)
            M_v[:, -1] = 0
            M_v[batched_NC_ids, batched_NC_ids] = 0
            M_v[batched_NC_ids, -1] = 1
            M_t = M_v.t()
            pi_v = self.sinkhorn_log_domain_torch(p, q, cost_v, Mask = M_v)
            pi_t = self.sinkhorn_log_domain_torch(q, p, cost_t, Mask = M_t)

            L_v = torch.zeros_like(M_v, device = options.device)
            L_v.diagonal().fill_(1)
            L_v[batched_NC_ids, batched_NC_ids] = 0
            L_v[batched_NC_ids, -1] = 1
            L_t= L_v.t()

            label_v2t = pi_v / (pi_v.sum(dim=1, keepdim=True) )
            label_v2t = gamma * label_v2t + (1 - gamma) * L_v

            label_t2v = pi_t / (pi_t.sum(dim=1, keepdim=True) )
            label_t2v = gamma * label_t2v + (1 - gamma) * L_t


        #Calculate loss manually using log-softmax and smoothed labels
        logits = torch.cat([logits_per_img, logits_paired_img_no], dim=1)
        log_probs_v2t = F.log_softmax(logits, dim=1)
        loss_img = self.kl_loss(log_probs_v2t, label_v2t)

        log_probs_t2v = F.log_softmax(logits.t(), dim=1)
        loss_txt = self.kl_loss(log_probs_t2v, label_t2v)
        loss_ul = (loss_img + loss_txt) / 2

        inmodal_opposite_loss = self.relation_opposite_loss(text_embeds, text_no_embeds)

        return loss_ul, inmodal_opposite_loss

