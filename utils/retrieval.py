import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "15"
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from PIL import Image, ImageFile
from pkgs.openai.clip import load as load_model
from pkgs.LaCLIP import models

# from pkgs.DeCLIP.clip import load as load_model
# from pkgs.DeCLIP import models


ImageFile.LOAD_TRUNCATED_IMAGES = True

def batch(iterable, n = 1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

@torch.no_grad()
def itm_eval(text_embeddings, image_embeddings):

    # sim_matrix_i2t = image_embeddings @ text_embeddings.t()
    # sim_matrix_t2i = text_embeddings @ image_embeddings.t()

    ## Image -> Text
    # ranks = np.zeros(len(sim_matrix_i2t))
    ranks = np.zeros(len(image_embeddings))

    for index in range(0, len(image_embeddings), 5):
        scores = image_embeddings[index] @ text_embeddings.t()
        # scores = sim_matrix_i2t[index]
        li = np.argsort(scores.detach().cpu().numpy())[::-1]
        for i in range(len(li)):
            if index <= li[i] and li[i] <= index + 4:
                rank = i
                break
        ranks[index] = rank

        # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    ## Image -> Text
    ranks = np.zeros(len(text_embeddings))
    for index in range(len(text_embeddings)):
        scores = text_embeddings[index] @ image_embeddings.t()
    # for index, scores in tqdm(enumerate(sim_matrix_t2i)):
        scores = scores[::5]
        li = np.argsort(scores.detach().cpu().numpy())[::-1]
        for i in range(len(li)):
            if li[i] == index//5:
                rank = i
                break
        ranks[index] = rank

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                'txt_r5': tr5,
                'txt_r10': tr10,
                'txt_r_mean': tr_mean,
                'img_r1': ir1,
                'img_r5': ir5,
                'img_r10': ir10,
                'img_r_mean': ir_mean,
                'r_mean': r_mean}

    return eval_result

def i2t_RCL(sims, return_ranks=False, img_div=5):
    """
    Images->Text (Image Annotation)
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # img_div = int(sims.shape[1] / sims.shape[0])

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]

        # Score
        rank = 1e20
        for i in range(img_div * index, img_div * index + img_div, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def t2i_RCL(sims, return_ranks=False, img_div=5):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # img_div = int(sims.shape[1] / sims.shape[0])

    npts = sims.shape[0]
    ranks = np.zeros(img_div * npts)
    top1 = np.zeros(img_div * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(img_div):
            inds = np.argsort(sims[img_div * index + i])[::-1]
            ranks[img_div * index + i] = np.where(inds == index)[0][0]
            top1[img_div * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

class TextImagePairDataset(Dataset):
    def __init__(self, texts, images):
        self.texts = texts
        self.images = images

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.images[idx]

def get_all_embeddings(model, all_texts, all_images, root, processor, batch_size = 1024, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose = False):
    text_embeddings = []
    image_embeddings = []

    with torch.no_grad():
        score = 0

        # dataloader_texts = list(batch(all_texts, batch_size))
        # dataloader_images = list(batch(all_images, batch_size))
        #
        # bar = zip(dataloader_texts, dataloader_images)
        # print("Evaluating..")
        # bar = tqdm(bar, total = len(dataloader_texts))


        # 创建 dataset 和 dataloader
        dataset = TextImagePairDataset(all_texts, all_images)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,  # 并行加载
            pin_memory=True,  # GPU 加速
            prefetch_factor=4  # 预加载因子
        )

        print("Evaluating..")
        bar = tqdm(dataloader, total=len(dataloader))

        for texts, images in bar:
            #captions = processor.process_text(texts)
            #input_ids = captions['input_ids'].to(device)
            #attention_mask = captions['attention_mask'].to(device)
            pixel_values = torch.tensor(np.stack([processor.process_image(Image.open(os.path.join(root, image)).convert("RGB")) for image in images])).to(device)

            text_embedding = model.get_text_features(texts, processor, device)
            #text_embedding = model.get_text_features(input_ids = input_ids, attention_mask = attention_mask)
            image_embedding = model.get_image_features(pixel_values)

            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)

            text_embeddings.append(text_embedding)
            image_embeddings.append(image_embedding)

        text_embeddings = torch.cat(text_embeddings)
        image_embeddings = torch.cat(image_embeddings)
        return text_embeddings, image_embeddings

def evaluate(input_file):

    if options.use_saved_embeddings:
        with open(options.embeddings_file, 'rb') as f:
            text_embeds, image_embeds = pickle.load(f)
        print('Embeddings Loaded!')
    else:
        #YFCC15M- NCU
        # processor = load_model()
        # model = getattr(models, 'CLIP_VITB32')(rand_embed=False)
        # ckpt_path = options.checkpoint
        # checkpoint = torch.load(ckpt_path, map_location='cpu')
        # state_dict = OrderedDict()
        # for k, v in checkpoint['state_dict'].items():
        #     state_dict[k.replace('module.', '')] = v
        # model.cuda()
        # model.load_state_dict(state_dict, strict=False)

        # YFCC15M- CLIP
        # processor = load_model()
        # model = getattr(models, 'CLIP_VITB32')(rand_embed=False)
        # ckpt_path = options.checkpoint
        # checkpoint = torch.load(ckpt_path, map_location='cpu')
        # model.cuda()
        # model.load_state_dict(checkpoint, strict=False)

        #CC3M/12M
        _, processor = load_model(name=options.model_name, pretrained=options.pretrained)
        model = getattr(models, 'CLIP_VITB32')(rand_embed=False)
        ckpt_path = options.checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        model.cuda()
        model.load_state_dict(state_dict, strict=False)


        model.eval()
        print(input_file)
        root = os.path.dirname(input_file)
        df = pd.read_csv(input_file, sep = options.delimiter)

        captions = df[options.caption_key].tolist()
        images = df[options.image_key].tolist()

        text_embeds, image_embeds = get_all_embeddings(model, captions, images, root = root, processor = processor, batch_size = options.batch_size, device = 'cuda')
        # torch.save(image_embeds, '/mnt/hanhc/CyCLIP/Features/ncu_img_embeddings.pt')
        # torch.save(text_embeds, '/mnt/hanhc/CyCLIP/Features/ncu_txt_embeddings.pt')
        # image_embeds = torch.load('/mnt/hanhc/CyCLIP/Features/img_embeddings.pt')
        # text_embeds = torch.load('/mnt/hanhc/CyCLIP/Features/txt_embeddings.pt')


        with open(options.embeddings_file, 'wb') as f:
            pickle.dump((text_embeds, image_embeds), f)
        print('Embedding dumped!')

    img_div = 5
    #假设 image_embeds 维度是 [5000, embed_dim]
    n_images_original = image_embeds.shape[0] // img_div  # 1000
    embed_dim = image_embeds.shape[1]
    # 重组 image_embeds，取每5个重复嵌入的最大值或平均值
    image_embeds_reformed = image_embeds.view(n_images_original, img_div, embed_dim)  # [1000, 5, embed_dim]
    #image_embeds_reformed = image_embeds_reformed.mean(dim=1)  # [1000, embed_dim]
    image_embeds_reformed = image_embeds_reformed.max(dim=1)[0]

    # 然后计算相似度
    #sims = image_embeds @ text_embeds.t()
    sims = image_embeds_reformed @ text_embeds.t()  # 得到 [1000, 5000] 的相似度矩阵
    sims = sims.cpu().numpy()
    # sims_list[-1].append(sims)
    r, rt = i2t_RCL(sims, return_ranks=True, img_div=img_div)
    ri, rti = t2i_RCL(sims, return_ranks=True, img_div=img_div)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    print("---------------------------------------")

    # result = itm_eval(text_embeds, image_embeds)
    #
    # print(result)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()

    #parser.add_argument("--input_file", type = str, default = '/mnt/hanhc/CyCLIP/data/MSCOCO/test/mscoco_test.csv', help = "Input file")
    parser.add_argument("--input_file", type=str,  default='/mnt/hanhc/CyCLIP/data/Flickr30K/test/flickr30k.csv', help="Input file")
    # parser.add_argument("-o,--output_file", dest = "output_file", type = str, default = None, help = "Output file")
    # parser.add_argument("-q,--quiet", dest = "quiet" , default = False, action = "store_true", help = "Silent output")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch Size")
    parser.add_argument("--delimiter", type = str, default = ",", help = "Input file delimiter")
    parser.add_argument("--image_key", type = str, default = "image", help = "Image column name")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "Caption column name")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")

    parser.add_argument("--checkpoint", default = '/mnt/hanhc/CyCLIP/logs/Rebuttal_VitB32_CC3M_Gamma_1.0/checkpoints/epoch_3.pt', type = str, help = "Path to checkpoint to resume training")

    #parser.add_argument("--checkpoint", default='/mnt/hanhc/CyCLIP/logs/best_unlearning_VitB32_CC12M_only_P/checkpoints/best_epoch_10.pt', type=str, help="Path to checkpoint to resume training")

    parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")
    parser.add_argument("--use_saved_embeddings", action = "store_true", default = False, help = "Use saved embeddings")
    parser.add_argument("--embeddings_file", type = str, default = "embeddings.pkl", help = "embedding file")


    options = parser.parse_args()
    evaluate(options.input_file)
