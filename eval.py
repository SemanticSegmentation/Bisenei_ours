from dataset.CamVid import CamVid
import torch
import argparse
import os
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import numpy as np
from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy
import tqdm
import cv2

def eval(model,dataset, args, label_info):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )
    print('start test!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('test')
        for i, (data, label) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            predict = model(data).squeeze()
            '''
            from PIL import Image
            import numpy as np
            
            temp = np.reshape(predict.detach().cpu().numpy(), (32, 640, 640))
            print(type(temp))
            temp = np.transpose(temp, [1, 2, 0])
            temp = np.asarray(temp[:, :])
            print(type(temp))
            for i in range(temp):
                for j in range(temp[0]):
                    k=max(j)
                    t=k.index()


            temp = np.asarray(temp < 0.05)
            new_im = Image.fromarray(temp)
            new_im.save('l.gif')
            print(predict)
            '''           
            predict = reverse_one_hot(predict)
            predict = colour_code_segmentation(np.array(predict.cpu()), label_info)
            #print(predict)
            #cv2.imwrite("./result/"+dataset.image_name[i]+"_R.png",predict)
            label = label.squeeze()
            label = reverse_one_hot(label)
            label = colour_code_segmentation(np.array(label.cpu()), label_info)

            precision = compute_global_accuracy(predict, label)
            precision_record.append(precision)
        precision = np.mean(precision_record)
        tq.close()
        print('precision for test: %.3f' % precision)
        return precision

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--crop_height', type=int, default=640, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped/resized input image to network')
    parser.add_argument('--data', type=str, default='/path/to/data', help='Path of training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    args = parser.parse_args(params)

    # create dataset and dataloader
    test_path = os.path.join(args.data, 'test')
    # test_path = os.path.join(args.data, 'train')
    test_label_path = os.path.join(args.data, 'test_labels')
    # test_label_path = os.path.join(args.data, 'train_labels')
    csv_path = os.path.join(args.data, 'class_dict.csv')
    dataset = CamVid(test_path, test_label_path, csv_path, scale=(args.crop_height, args.crop_width), mode='test')


    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    # get label info
    label_info = get_label_info(csv_path)
    # test
    eval(model, dataset , args, label_info)


if __name__ == '__main__':
    params = [
        '--checkpoint_path', './checkpoints/epoch_295.pth',
        #'--data', '/path/to/CamVid',
        '--data', '/media/charel/CCC86472C8645CA8/BiSeNet-master/CamVid',
        '--cuda', '0'
    ]
    main(params)
