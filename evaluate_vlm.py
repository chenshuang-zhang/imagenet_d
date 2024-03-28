
import torch
import argparse
import os
import json
from tqdm import tqdm
from utils.data_loaders_imgnet_d_id import ImageNetDIDLoader
import clip
import csv

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluate vision language models, e.g., CILP', add_help=False)
    parser.add_argument('--model', default='ViT-B/16', type=str, metavar='MODEL', help='Name of model to test')
    return parser

def test_model(test_loader_dict, model, args):
    print(f"Test {args.model}\n")

    model.eval()

    with open('preprocessing/imgnet_d_dir2imgnet_d_id.txt') as f:
        category_mapping = json.load(f)
        sorted_categories = sorted(category_mapping.values(), key=lambda value: value[0])
        category_list = [value[1].replace('_', ' ').replace('-', ' ').replace('/', ' or ').lower() for value in sorted_categories]
        id2category = {key: value for key, value in enumerate(category_list)}

    text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in category_list]).to("cuda")

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    os.makedirs("result/", exist_ok=True)
 
    for datatype, test_loader in test_loader_dict.items():
        pbar_prefix = 'Testing on ' + str(datatype)
        top1_cnt = 0
        top5_cnt = 0
        cnt = 0

        result_dir = f"result/{datatype}/"
        os.makedirs(result_dir, exist_ok=True)
        logname = f"{result_dir}/{args.model.replace('/','-')}_scores.csv"
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter='\t')
            logwriter.writerow(["image_dir", "gt_category", "top1_pred_cate", "top2_pred_cate"])

        with torch.no_grad():
            for sample, example in tqdm(enumerate(test_loader), total=len(test_loader), desc=pbar_prefix):
                images = example['images']
                target = example['labels']
                path = example['path']
   
                images = images.cuda()

                clip_image_features = model.encode_image(images)
                clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
                output = (100.0 * clip_image_features @ text_features.T).softmax(dim=-1)
 
                maxk = 5
                batch_size = len(target)

                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()

                correct = pred.eq(target.cuda().expand_as(pred))
                top1_cnt += correct[:1].view(-1).float().sum(0, keepdim=True)
                top5_cnt += correct[:5].reshape(-1).float().sum(0, keepdim=True)

                cnt += batch_size

                # Record the predictions 
                with open(logname, 'a+') as logfile:
                    logwriter = csv.writer(logfile, delimiter='\t')
                    for img_idx in range(batch_size):
                        logwriter.writerow(["/".join(path[img_idx].split('/')[-3:]),
                                            id2category[target[img_idx].item()],
                                            id2category[pred[0,img_idx].item()],
                                            id2category[pred[1,img_idx].item()]
                        ])

            print("Test accuracy: %d images top 1: %.5f, top 5: %.5f" %
                  (cnt, top1_cnt * 100.0 / cnt, top5_cnt * 100.0 / cnt))
    
        with open(f"{result_dir}/clip_accuracy.txt", 'a+') as txt_file:
            txt_file.write(f'{args.model}\t{cnt}\t{top1_cnt.item() * 100.0 / cnt}\t{top5_cnt.item() * 100.0 / cnt}\n')

            
def test():
    args = get_args_parser()
    args = args.parse_args()

    imagenet_d_dir = './data/ImageNet-D/'

    model, test_transform = clip.load(args.model, "cuda", jit=False)
    model.eval()

    test_loader_all = dict()
    subset_list = ['background', 'texture', 'material']
    for subset in subset_list:
        test_loader = torch.utils.data.DataLoader(
            ImageNetDIDLoader(os.path.join(imagenet_d_dir, subset), transform=test_transform),
            batch_size=100, shuffle=True,
            num_workers=10, pin_memory=True)
        test_loader_all[subset] = test_loader

    test_model(test_loader_all, model, args)


if __name__ == "__main__":
    test()