import cv2
import torch
import tqdm
import os
import numpy as np
import pickle
import time
import argparse
import importlib

from data.coco_pose.ref import ref_dir, flipRef
from utils.misc import get_transform, kpt_affine, resize
from utils.group import HeatmapParser

'''
test_demo.py是在test.py基础上修改的，解决了test.py不能检测单张图像的问题。
test_demo.py将原来test.py引用train.py中初始化init等函数变成不依赖train.py（通过将函数搬运过来修改）；
并且支持命令行解析输入图像和输出图像
'''

# valid_filepath = ref_dir + '/validation.pkl'

H_parser = HeatmapParser(detection_val=0.1)

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_image_path', type=str,default='pangzi.jpg', help='input image name')
    parser.add_argument('-o', '--output_image_path', type=str, default='output.jpg', help='output image name')
    parser.add_argument('-t', '--task', type=str, default='pose', help='task to be trained')    
    parser.add_argument('-c', '--continue_exp', type=str, default='pretrained',help='continue exp') # 调试时使用
    # parser.add_argument('-c', '--continue_exp', type=str,help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='pose', help='experiments name')
    parser.add_argument('-m', '--mode', type=str, default='single', help='scale mode')
    args = parser.parse_args()
    return args

def reload(config):
    """
    load or initialize model's parameters by config from config['opt'].continue_exp
    config['train']['epoch'] records the epoch num
    config['inference']['net'] is the model
    """
    opt = config['opt']

    if opt.continue_exp:
        resume = os.path.join('exp', opt.continue_exp)
        resume_file = os.path.join(resume, 'checkpoint.pth.tar')
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file) #加载模型

            config['inference']['net'].load_state_dict(checkpoint['state_dict'])
            config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            config['train']['epoch'] = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            exit(0)

    if 'epoch' not in config['train']:
        config['train']['epoch'] = 0

def refine(det, tag, keypoints):
    """
    Given initial keypoint predictions, we identify missing joints
    """
    if len(tag.shape) == 3:
        tag = tag[:,:,:,None]

    tags = []
    for i in range(keypoints.shape[0]):
        if keypoints[i, 2] > 0:
            y, x = keypoints[i][:2].astype(np.int32)
            tags.append(tag[i, x, y])

    prev_tag = np.mean(tags, axis = 0)
    ans = []

    for i in range(keypoints.shape[0]):
        tmp = det[i, :, :]
        tt = (((tag[i, :, :] - prev_tag[None, None, :])**2).sum(axis = 2)**0.5 )
        tmp2 = tmp - np.round(tt)

        x, y = np.unravel_index( np.argmax(tmp2), tmp.shape )
        xx = x
        yy = y
        val = tmp[x, y]
        x += 0.5
        y += 0.5

        if tmp[xx, min(yy+1, det.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
            y+=0.25
        else:
            y-=0.25

        if tmp[min(xx+1, det.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
            x+=0.25
        else:
            x-=0.25

        x, y = np.array([y,x])
        ans.append((x, y, val))
    ans = np.array(ans)

    if ans is not None:
        for i in range(17):
            if ans[i, 2]>0 and keypoints[i, 2]==0:
                keypoints[i, :2] = ans[i, :2]
                keypoints[i, 2] = 1 

    return keypoints

def multiperson(img, func, mode):
    """
    1. Resize the image to different scales and pass each scale through the network
    2. Merge the outputs across scales and find people by HeatmapParser
    3. Find the missing joints of the people with a second pass of the heatmaps
    """
    if mode == 'multi':
        scales = [2, 1., 0.5]
    else:
        scales = [1]

    height, width = img.shape[0:2]
    center = (width/2, height/2)
    dets, tags = None, []
    for idx, i in enumerate(scales):
        scale = max(height, width)/200
        input_res = max(height, width)
        inp_res = int((i * 512 + 63)//64 * 64)
        res = (inp_res, inp_res)  #resize后的图像分辨率、尺寸大小

        mat_ = get_transform(center, scale, res)[:2]
        inp = cv2.warpAffine(img, mat_, res)/255

        def array2dict(tmp):
            return {
                'det': tmp[0][:,:,:17],
                'tag': tmp[0][:,-1, 17:34]
            }

        tmp1 = array2dict(func([inp]))
        tmp2 = array2dict(func([inp[:,::-1]]))

        tmp = {}
        for ii in tmp1:
            tmp[ii] = np.concatenate((tmp1[ii], tmp2[ii]),axis=0)

        det = tmp['det'][0, -1] + tmp['det'][1, -1, :, :, ::-1][flipRef]
        if det.max() > 10:
            continue
        if dets is None:
            dets = det
            mat = np.linalg.pinv(np.array(mat_).tolist() + [[0,0,1]])[:2]
        else:
            dets = dets + resize(det, dets.shape[1:3]) 

        if abs(i-1)<0.5:
            res = dets.shape[1:3]
            tags += [resize(tmp['tag'][0], res), resize(tmp['tag'][1,:, :, ::-1][flipRef], res)]

    if dets is None or len(tags) == 0:
        return [], []

    tags = np.concatenate([i[:,:,:,None] for i in tags], axis=3)
    dets = dets/len(scales)/2
    
    dets = np.minimum(dets, 1)
    grouped = H_parser.parse(np.float32([dets]), np.float32([tags]))[0]


    scores = [i[:, 2].mean() for  i in grouped]

    for i in range(len(grouped)):
        grouped[i] = refine(dets, tags, grouped[i])

    if len(grouped) > 0:
        grouped[:,:,:2] = kpt_affine(grouped[:,:,:2] * 4, mat)
    
    # 筛选并整合人体关键点信息 此处依据pose-ae-demo修改
    persons = []
    for val in grouped: # val为某一个人的关键点信息
        if val[:, 2].max()>0: # 某个人的17个关键点中最大的prediction必须大于0
            tmp = {"keypoints": [], "score":float(val[:, 2].mean())}  # 将17个关键点的平均值作为score分数值
            for j in val:  # j表示17个关键点中的某一个
                if j[2]>0.: # 关键点的prediction必须大于0，否则认为检测错误，记为[0,0,0]
                    tmp["keypoints"]+=[float(j[0]), float(j[1]), float(j[2])]
                else:
                    tmp["keypoints"]+=[0, 0, 0]
            persons.append(tmp)
    # return persons # 返回满足要求的所有人
    return persons, grouped, scores

def coco_eval(prefix, dt, gt):
    """
    Evaluate the result with COCO API
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    for _, i in enumerate(sum(dt, [])):
        i['id'] = _+1

    image_ids = []
    import copy
    gt = copy.deepcopy(gt)

    dic = pickle.load(open(valid_filepath, 'rb'))
    paths, anns, idxes, info = [dic[i] for i in ['path', 'anns', 'idxes', 'info']]

    widths = {}
    heights = {}
    for idx, (a, b) in enumerate(zip(gt, dt)):
        if len(a)>0:
            for i in b:
                i['image_id'] = a[0]['image_id']
            image_ids.append(a[0]['image_id'])
        if info[idx] is not None:
            widths[a[0]['image_id']] = info[idx]['width']
            heights[a[0]['image_id']] = info[idx]['height']
        else:
            widths[a[0]['image_id']] = 0
            heights[a[0]['image_id']] = 0
    image_ids = set(image_ids)

    import json
    cat = [{'supercategory': 'person', 'id': 1, 'name': 'person', 'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]], 'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']}]
    with open(prefix + '/gt.json', 'w') as f:
        json.dump({'annotations':sum(gt, []), 'images':[{'id':i, 'width': widths[i], 'height': heights[i]} for i in image_ids], 'categories':cat}, f)

    with open(prefix + '/dt.json', 'w') as f:
        json.dump(sum(dt, []), f)

    coco = COCO(prefix + '/gt.json')
    coco_dets = coco.loadRes(prefix + '/dt.json')
    coco_eval = COCOeval(coco, coco_dets, "keypoints")
    coco_eval.params.imgIds = list(image_ids)
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

def genDtByPred(pred, image_id = 0):
    """
    Generate the json-style data for the output 
    """
    ans = []
    for i in pred:
        val = pred[i] if type(pred) == dict else i
        if val[:, 2].max()>0:
            tmp = {'image_id':int(image_id), "category_id": 1, "keypoints": [], "score":float(val[:, 2].mean())}
            p = val[val[:, 2]> 0][:, :2].mean(axis = 0)
            for j in val:
                if j[2]>0.:
                    tmp["keypoints"] += [float(j[0]), float(j[1]), 1]
                else:
                    tmp["keypoints"] += [float(p[0]), float(p[1]), 1]
            ans.append(tmp)
    return ans

def get_img(inp_res = 512):
    """
    Load validation images
    """
    if os.path.exists(valid_filepath) is False:
        from utils.build_valid import main #如果不存在测试图像数据集，则创建
        main()

    dic = pickle.load(open(valid_filepath, 'rb'))
    paths, anns, idxes, info = [dic[i] for i in ['path', 'anns', 'idxes', 'info']]

    total = len(paths)
    tr = tqdm.tqdm( range(0, total), total = total )
    for i in tr:
        img = cv2.imread(paths[i])[:,:,::-1]
        yield anns[i], img

# 一共17个关键点
part_labels = ['nose','eye_l','eye_r','ear_l','ear_r',
               'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
               'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r']
part_idx = {b:a for a, b in enumerate(part_labels)}  # 生成字典'nose':0等

def draw_limbs(inp, pred):
    """
    inp:input image
    pred:检测到的人体关键点，坐标x1,y1,prediction1,x2,y2,prediction2...x17,y17,prediction17
    """
    def link(a, b, color):
        """设置关键点连线的属性"""
        if part_idx[a] < pred.shape[0] and part_idx[b] < pred.shape[0]:
            a = pred[part_idx[a]]
            b = pred[part_idx[b]]
            # a,b 为某个关键点的信息，包括x,y,prediction
            if a[2]>0.07 and b[2]>0.07: # 只有当关键点的可能性高于0.07时，才连接关键点
                cv2.line(inp, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 4) # 线条粗细为6

    pred = np.array(pred).reshape(-1, 3)
    bbox = pred[pred[:,2]>0]
    a, b, c, d = bbox[:,0].min(), bbox[:,1].min(), bbox[:,0].max(), bbox[:,1].max()

    # 绘制一个包括17个人体关键点的最小边框，表示检测到的某个人
    cv2.rectangle(inp, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 2) # 白色边框

    # 这里定义了，17个关键点之间如何连接，以及线条颜色
    link('nose', 'eye_l', (255, 0, 0))
    link('eye_l', 'eye_r', (255, 0, 0))
    link('eye_r', 'nose', (255, 0, 0))

    link('eye_l', 'ear_l', (255, 0, 0))
    link('eye_r', 'ear_r', (255, 0, 0))

    link('ear_l', 'sho_l', (255, 0, 0))
    link('ear_r', 'sho_r', (255, 0, 0))
    link('sho_l', 'sho_r', (255, 0, 0))
    link('sho_l', 'hip_l', (0, 255, 0))
    link('sho_r', 'hip_r',(0, 255, 0))
    link('hip_l', 'hip_r', (0, 255, 0))

    link('sho_l', 'elb_l', (0, 0, 255))
    link('elb_l', 'wri_l', (0, 0, 255))

    link('sho_r', 'elb_r', (0, 0, 255))
    link('elb_r', 'wri_r', (0, 0, 255))

    link('hip_l', 'kne_l', (255, 255, 0))
    link('kne_l', 'ank_l', (255, 255, 0))

    link('hip_r', 'kne_r', (255, 255, 0))
    link('kne_r', 'ank_r', (255, 255, 0))


def test_init():
    '''根据train.py中的init修改而来'''
    opt = parse_command_line()
    # import_module 类似与Python关键字引入模型：import **。优点是带参数，可以在运行过程中引入某一模型
    task = importlib.import_module('task.' + opt.task) #载入模型配置文件 task/pose.py
    config = task.__config__
    config['opt'] = opt  # 给config增加一项'opt'
    func = task.make_network(config)
    reload(config)
    return func, config

def main():
    # from train import init # 引入train.py中的命令行解析函数等初始化
    tic=time.time()
    func, config = test_init()
    toc=time.time()
    print("init time cost:{}".format(toc-tic)) #加载模型，初始化时间
    mode = config['opt'].mode #single默认 multi

    def runner(imgs):
        return func(0, config, 'inference', imgs=torch.Tensor(np.float32(imgs)))['preds']

    def do(img): # 检测单张图像
        tic=time.time()
        persons, ans, scores = multiperson(img, runner, mode)
        toc=time.time()
        print("Detection time cost:{}".format(toc-tic)) #检测时间
        if len(ans) > 0:
            ans = ans[:,:,:3]

        pred = genDtByPred(ans)

        # draw 绘制人体姿态关键点和边框
        # img=np.array(img)      
        for i in persons:
            draw_limbs(img, i["keypoints"])
        # cv2.imwrite('tmp.jpg', img)

        for i, score in zip( pred, scores ):
            i['score'] = float(score)
        return pred

    #测试单张图像
    img = cv2.imread(config['opt'].input_image_path)
    pred=do(img)
    print("------Detection Keypoints--------")
    import pprint
    pprint.pprint(pred)
    # pprint(pred)
    cv2.imwrite(config['opt'].output_image_path, img)

    # gts = [] # groundtrues
    # preds = []

    # idx = 0
    # for anns, img in get_img(inp_res=-1): #inp_res是输入分辨率大小，长=宽
    #     idx += 1
    #     gts.append(anns)
    #     preds.append(do(img))

    # prefix = os.path.join('exp', config['opt'].exp) #exp/pose
    # coco_eval(prefix, preds, gts)

if __name__ == '__main__':
    main()
