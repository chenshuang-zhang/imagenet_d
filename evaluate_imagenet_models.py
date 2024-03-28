
import torch
import argparse
from utils.models import load_model
import os
from tqdm import tqdm
from utils.data_loaders_imgnet_id import ImageNetDLoader

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluate vision models trained on ImageNet', add_help=False)
    parser.add_argument('--model', default='vgg19', type=str, metavar='MODEL', help='Name of model to test')
    return parser

def test_model(test_loader_dict, model, args):
    model.eval()
    
    for datatype, test_loader in test_loader_dict.items():
        top1_cnt = 0
        top5_cnt = 0
        cnt = 0

        result_dir = f"result/{datatype}/"
        os.makedirs(result_dir, exist_ok=True)
        pbar_prefix = 'Testing on ' + str(datatype)

        with torch.no_grad():
            for sample, example in tqdm(enumerate(test_loader), total=len(test_loader), desc=pbar_prefix):
                images = example['images']
                target = example['labels']

                images = images.cuda()

                output = model(images)
                maxk = 5
                batch_size = target[0].shape[0]
          
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()

                for k in range(len(target)):
                    correct = pred.eq(target[k].cuda().view(1, -1).expand_as(pred))
                    top1_cnt += correct[:1].view(-1).float().sum(0, keepdim=True)
                    top5_cnt += correct[:5].reshape(-1).float().sum(0, keepdim=True)

                cnt += batch_size

            print("Test accuracy:  %d images top 1: %.5f, top 5: %.5f" %
                  (cnt, top1_cnt * 100.0 / cnt, top5_cnt * 100.0 / cnt))
            with open(f"{result_dir}/imagenet_model_accuracy.txt", 'a+') as txt_file:
                txt_file.write(f'{args.model}\t{cnt}\t{top1_cnt.item() * 100.0 / cnt}\t{top5_cnt.item() * 100.0 / cnt}\n')

def test():
    args = get_args_parser()
    args = args.parse_args()

    imagenet_d_dir = './data/ImageNet-D/'

    model, test_transform = load_model(args.model)
    model.cuda()
    model.eval()

    test_loader_all = dict()
    subset_list = ['background', 'texture', 'material']
    for subset in subset_list:
        test_loader = torch.utils.data.DataLoader(
            ImageNetDLoader(os.path.join(imagenet_d_dir, subset), transform=test_transform),
            batch_size=100, shuffle=True,
            num_workers=10, pin_memory=True)
        test_loader_all[subset] = test_loader

    test_model(test_loader_all, model, args)


if __name__ == "__main__":
    test()