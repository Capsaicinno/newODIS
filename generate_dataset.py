import torch
import numpy as np
from dataset import Sun360PreExtractDataset
from torchvision.transforms.functional import to_pil_image
import tqdm
import os
from utils.util import to_code,code_to_img,unicode_to_img
from PIL import Image
device = "cpu"

vqvaes = torch.load("vqvae_models.pth")
vqvaes = [i.to(device) for i in vqvaes]
encoder,decoder,quantizer = [i.eval() for i in vqvaes]
img_path = "../sun360_one" if os.path.exists("../sun360_one") else "../../sun360_one"
dataset = Sun360PreExtractDataset(img_path, train=True,extract=True)
dataset_test = Sun360PreExtractDataset(img_path, train=False,extract=False)
os.makedirs("datas_all_one",exist_ok=True)
os.makedirs("datas_all_test_one",exist_ok=True)
os.makedirs("vqvae_recon_one",exist_ok=True)
os.makedirs("vqvae_recon_one_test",exist_ok=True)

torch.backends.cudnn.benchmark=True

with torch.inference_mode():
    for i,d in enumerate(tqdm.tqdm(dataset)):
        data,path,roll_pixel = d
        #data = img = torch.from_numpy(cv2.resize(img,(512,256))).permute(2,0,1)
        data = to_code(data,encoder,quantizer,device)
        np.savez_compressed(f"datas_all_one/{i:06}.npz",data=data,path=np.array([path]),degree=np.array([roll_pixel]))
        if roll_pixel==0:
            img = to_pil_image(code_to_img(data,quantizer,decoder,device)[0])
            img.save(f"vqvae_recon_one/{os.path.basename(path)}")

#original
# with torch.inference_mode():
#     for i,d in enumerate(tqdm.tqdm(dataset_test)):
#         data,path,roll_pixel = d
#         #data = img = torch.from_numpy(cv2.resize(img,(512,256))).permute(2,0,1)
#         data = to_code(data,encoder,quantizer,device)
#         np.savez_compressed(f"datas_all_test_one/{i:06}.npz",data=data,path=np.array([path]),degree=np.array([roll_pixel]))
#         if roll_pixel==0:
#             img = to_pil_image(code_to_img(data,quantizer,decoder,device)[0])
#             img.save(f"vqvae_recon_one_test/{os.path.basename(path)}")

###############PATTERN_1   test用データ　エンコード前に画像の左上を白に加工##########################
# with torch.inference_mode():
#     for i,d in enumerate(tqdm.tqdm(dataset_test)):
#         data,path,roll_pixel = d
#         patch_size = 128
#         data[:,:patch_size, :patch_size] = 1
#         #data = img = torch.from_numpy(cv2.resize(img,(512,256))).permute(2,0,1)
#         data = to_code(data,encoder,quantizer,device)
#         np.savez_compressed(f"datas_all_test_one/{i:06}.npz",data=data,path=np.array([path]),degree=np.array([roll_pixel]))
#         if roll_pixel==0:
#             img = to_pil_image(code_to_img(data,quantizer,decoder,device)[0])
#             img.save(f"vqvae_recon_one_test/{os.path.basename(path)}")


# ###############PATTERN_2   test用データ　エンコーダ後にコードブックの左上を白に加工#####################
# with torch.inference_mode():
#     for i,d in enumerate(tqdm.tqdm(dataset_test)):
#         data,path,roll_pixel = d
#         #data = img = torch.from_numpy(cv2.resize(img,(512,256))).permute(2,0,1)
#         data = to_code(data,encoder,quantizer,device)
        
#         #codebookの中身を変更
#         patch_size = 4
#         max_idx = data.max()
#         data = data.reshape(16,32)#flattenされていた部分を長方形型に
#         data[:patch_size,:patch_size] = max_idx
#         data = data.flatten()#flatに戻す
        

#         np.savez_compressed(f"datas_all_test_one/{i:06}.npz",data=data,path=np.array([path]),degree=np.array([roll_pixel]))
#         if roll_pixel==0:
#             img = to_pil_image(code_to_img(data,quantizer,decoder,device)[0])
#             img.save(f"vqvae_recon_one_test/{os.path.basename(path)}")

###############PATTERN_3  コードブックを分けて別々で生成　その後画像を並べて結合　##############################
with torch.inference_mode():
    for i,d in enumerate(tqdm.tqdm(dataset_test)):
        data,path,roll_pixel = d
        #data = img = torch.from_numpy(cv2.resize(img,(512,256))).permute(2,0,1)
        data = to_code(data,encoder,quantizer,device)
        np.savez_compressed(f"datas_all_test_one/{i:06}.npz",data=data,path=np.array([path]),degree=np.array([roll_pixel]))
        if roll_pixel==0:
            concat_img = Image.new('RGB',(512,256))
            for i in range(16):
                for j in range(32):
                    code = data[32*i+j]
                    img_patch = to_pil_image(unicode_to_img(code,quantizer,decoder,device)[0])
                    concat_img.paste(img_patch,(16*j,16*i))
            concat_img.save(f"vqvae_recon_one_test/{os.path.basename(path)}")