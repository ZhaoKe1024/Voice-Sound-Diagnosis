#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/5/23 14:06
# @Author: ZhaoKe
# @File : infer.py
# @Software: PyCharm
import numpy as np
import librosa
import torch
from cls_model import CLSModel
from tools import segment_cough, vad
label_mapping = {
    0: 'Healthy(健康样本)',
    1: 'Hyperkinetic Dysphonia(声带运动过度或不协调)',
    2: 'Reflux Laryngitis(反流性喉炎)',
    3: 'symptomic()',
}
LENGTH = 3872
sr = 8000

def load_ckpt():
    model = CLSModel()
    model.load_state_dict(torch.load(f"./ckpt/cls_model_40.pt"))
    return model


def infer_one(wav_path, model):
    wav1, sr = librosa.load(wav_path)
    segs1, _ = segment_cough(x=wav1, fs=sr)
    if len(segs1) < 2:
        wav1 = vad(wav1, top_db=20, overlap=40)
        if len(wav1) < LENGTH:
            inp_wav = np.zeros(LENGTH)
            st = (LENGTH - len(wav1)) // 2
            inp_wav[st:st + len(wav1)] = wav1
            inp_wavs = [inp_wav]
        elif len(wav1) > LENGTH:
            inp_wavs = []
            overlap = 128
            st = 0
            while st+LENGTH < len(wav1):
                inp_wavs.append(wav1[st:st+LENGTH])
                st = st + LENGTH - overlap
            # st = (len(wav1)-LENGTH) // 2
            # inp_wav = wav1[st:st+LENGTH]
        else:
            inp_wavs = [wav1]
        inp_mfccs = []
        for inp_wav in inp_wavs:
            tmp = torch.from_numpy(librosa.feature.mfcc(y=inp_wav, sr=sr, n_mfcc=26))
            # print(tmp.shape)
            inp_mfccs.append(tmp)
        # print(len(inp_mfccs))
        # print(torch.stack(inp_mfccs, dim=0).shape)
        inp_mel = torch.stack(inp_mfccs, dim=0)
        print(inp_mel.shape)
        # pred = model(inp_wav=inp_wav)
        # print(len(segs1))
        # for item in segs1:
        #     print(item.shape)

    else:
        pass

    pred = model(inp_wav=inp_mel.transpose(1, 2))
    pred_cls = pred.argmax(-1).data.cpu().numpy()
    if len(pred_cls) == 1:
        pred_name = label_mapping[pred_cls[0]]
        print("Result:", pred_cls, pred_name)
    else:
        print(pred_cls)
        res_idx = np.argmax(np.bincount(pred_cls))
        print("Result:", res_idx, label_mapping[res_idx])



def infer_main():
    cls_model = load_ckpt()
    infer_one(wav_path="./datasets/test_xmj.m4a", model=cls_model)
    # infer_one(wav_path="./datasets/test_gsy.m4a", model=cls_model)
    # infer_one(wav_path="./datasets/test_zk.m4a", model=cls_model)


if __name__ == '__main__':
    infer_main()
