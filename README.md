# Do Inpainting Yourself: Generative Facial Inpainting Guided by Exemplars (EXE-GAN)
Official PyTorch implementation of EXE-GAN.
[[Homepage]](https://longlongaaago.github.io/EXE-GAN/)
[[paper]](https://arxiv.org/abs/2202.06358)
[[demo_youtube]](https://www.youtube.com/watch?v=nNEc94hgjtk)
[[demo_bilibili]](https://www.bilibili.com/video/BV14V4y1s7rz/?share_source=copy_web&vd_source=6fb8e0068d30286602ee8ea389f82ce4)

<div style="text-align: justify"> We present EXE-GAN, a novel exemplar-guided facial inpainting framework using generative adversarial networks. Our
approach can not only preserve the quality of the input facial image but also complete the image with exemplar-like facial attributes.</div>

![Performance](./imgs/teaser.png)


## Notice
Our paper was first released on Sun, 13 Feb 2022. 
We are thankful for the community's recognition and attention to our project.
We also recognized that there have been some great papers published after ours,
and we encourage you to check out their projects as well:
- [Paint by Example](https://arxiv.org/abs/2211.13227), [codes](https://github.com/Fantasy-Studio/Paint-by-Example) (released at Wed, 23 Nov 2022, CVPR 2023)
- [Reference-Guided Face Inpainting](https://arxiv.org/abs/2303.07014), [codes](https://github.com/wuyangluo/reffaceinpainting) (released at Mon, 13 Mar 2023, TCSVT 2023)
- [PATMAT](https://arxiv.org/abs/2304.06107), [codes](https://github.com/humansensinglab/PATMAT) (released at Wed, 12 Apr 2023, ICCV 2023)

## Requirements 
```
cd EXE-GAN project
pip install -r requirements.txt
```
- Note that other versions of PyTorch (e.g., higher than 1.7) also work well, but you have to install the corresponding CUDA version. 

##### What we have released
- [x] Training and testing codes
- [x] Pre-trained models

## Training
- Prepare your dataset (download [FFHQ](https://github.com/NVlabs/ffhq-dataset), and [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans))
- The folder structure of training and testing data is shown below:  
```
root/
    test/
        xxx.png
        ...
        xxz.png
    train/
        xxx.png
        ...
        xxz.png
```
- Prepare pre-trained checkpoints:
[Arcface.pth](https://drive.google.com/file/d/18w_YKb0cLX6LAdY4008vEgCPD-_3RmRE/view?usp=drive_link) and 
[psp_ffhq_encode.pt](https://drive.google.com/file/d/1_GdbsT1A5dyxF0FqOEiFlmouVsyf7Ag1/view?usp=drive_link) (put models in ./pre-train)


- Training
> python train.py --path /root/train --test_path /root/test
--size 256 --embedding_weight 0.1 --id_loss_weight 0.1 --percept_loss_weight 0.5 --arcface_path ./pre-train/Arcface.pth
--psp_checkpoint_path ./pre-train/psp_ffhq_encode.pt

## Testing 
### Note 
- For editing images from the web, photos are aligned by face landmarks and cropped to 256x256 by [align_face](https://github.com/ZPdesu/Barbershop/blob/main/align_face.py).

- [Irregular masks](https://nv-adlr.github.io/publication/partialconv-inpainting) (optional, if you would like to test on irregular masks, download Testing Set masks)
- (use our FFHQ_60k pre-trained model [EXE_GAN_model.pt](https://drive.google.com/file/d/1y7ThKBXL7QK7CPtvT3KICeNOu1T2xlCA/view?usp=drive_link) or trained *pt file by yourself.)
> python test.py --path /root/test  --size 256 --psp_checkpoint_path ./pre-train/psp_ffhq_encode.pt --ckpt ./checkpoint/EXE_GAN_model.pt
--mask_root ./dataset/mask/testing_mask_dataset
--mask_file_root ./dataset/mask
--mask_type test_6.txt

```
- mask_root Irregular masks root
- mask_file_root file name list file folder
- mask_type could be ["center", "test_2.txt", "test_3.txt", "test_4.txt", "test_5.txt", "test_6.txt", "all"]
```
- If you don't have irregular masks, just using center masks is also fine.
> python test.py --path /root/test  --size 256 --psp_checkpoint_path ./pre-train/psp_ffhq_encode.pt --ckpt ./checkpoint/EXE_GAN_model.pt
--mask_type center



## Exemplar-guided facial image recovery 
### Note 
- For editing images from the web, photos are aligned by face landmarks and cropped to 256x256 by [align_face](https://github.com/ZPdesu/Barbershop/blob/main/align_face.py).

(use our FFHQ_60k pre-trained model [EXE_GAN_model.pt](https://drive.google.com/file/d/1y7ThKBXL7QK7CPtvT3KICeNOu1T2xlCA/view?usp=drive_link) or trained *pt file by yourself.)
> python guided_recovery.py --psp_checkpoint_path ./pre-train/psp_ffhq_encode.pt
--ckpt  ./checkpoint/EXE_GAN_model.pt  --masked_dir ./imgs/exe_guided_recovery/mask --gt_dir ./imgs/exe_guided_recovery/target --exemplar_dir ./imgs/exe_guided_recovery/exemplar --sample_times 10
> --eval_dir ./recover_out  
```
- masked_dir: mask input folder
- gt_dir: the input gt_dir, used for  editing 
- exemplar_dir: exemplar_dir, the exemplar dir, for guiding the editing
- eval_dir: output dir
```
| <img src="./imgs/exe_guided_recovery/target/1_real.png"  height=180 width=180 alt="Ground-truth"> | <img src="./imgs/exe_guided_recovery/mask/1_mask.png" width=180 height=180 alt="Masked "> | <img src="./imgs/exe_guided_recovery/exemplar/1_exe.png" height=180 width=180 alt=" "> |<img src="./imgs/exe_guided_recovery/recover_out/1_inpaint.png" height=180 width=180 alt=" "> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |:----------------------------------------------------------: |
| <img src="./imgs/exe_guided_recovery/target/2_real.png"  height=180 width=180 alt="Ground-truth"> | <img src="./imgs/exe_guided_recovery/mask/2_mask.png" width=180 height=180 alt="Masked "> | <img src="./imgs/exe_guided_recovery/exemplar/2_exe.png" height=180 width=180 alt=" "> |<img src="./imgs/exe_guided_recovery/recover_out/2_inpaint.png" height=180 width=180 alt=" "> |
| <img src="./imgs/exe_guided_recovery/target/3_real.png"  height=180 width=180 alt="Ground-truth"> | <img src="./imgs/exe_guided_recovery/mask/3_mask.png" width=180 height=180 alt="Masked "> | <img src="./imgs/exe_guided_recovery/exemplar/3_exe.png" height=180 width=180 alt=" "> |<img src="./imgs/exe_guided_recovery/recover_out/3_inpaint.png" height=180 width=180 alt=" "> |
| <img src="./imgs/exe_guided_recovery/target/4_real.png"  height=180 width=180 alt="Ground-truth"> | <img src="./imgs/exe_guided_recovery/mask/4_mask.png" width=180 height=180 alt="Masked "> | <img src="./imgs/exe_guided_recovery/exemplar/4_exe.png" height=180 width=180 alt=" "> |<img src="./imgs/exe_guided_recovery/recover_out/4_inpaint.png" height=180 width=180 alt=" "> |
|          Ground-truth                        |                      Mask                               | Exemplar       | Inpainted  | 

- Inherent diversity, set ``--sample_times 10``  higher to get more diverse results.

| <img src="./imgs/exe_guided_recovery/diversity/1_0_inpaint.png"  height=180 width=180 alt="Ground-truth"> | <img src="./imgs/exe_guided_recovery/diversity/1_1_inpaint.png" width=180 height=180 alt="Masked "> | <img src="./imgs/exe_guided_recovery/diversity/1_2_inpaint.png" height=180 width=180 alt=" "> |<img src="./imgs/exe_guided_recovery/diversity/1_3_inpaint.png" height=180 width=180 alt=" "> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |:----------------------------------------------------------: |
|          diversity 1                       |                      diversity 2                               | diversity 3       | diversity 4  | 



## Exemplar guided style mixing 
### Note 
- For editing images from the web, photos are aligned by face landmarks and cropped to 256x256 by [align_face](https://github.com/ZPdesu/Barbershop/blob/main/align_face.py).

(use our FFHQ_60k pre-trained model [EXE_GAN_model.pt](https://drive.google.com/file/d/1y7ThKBXL7QK7CPtvT3KICeNOu1T2xlCA/view?usp=drive_link) or trained *pt file by yourself.)
> python exemplar_style_mixing.py --psp_checkpoint_path ./pre-train/psp_ffhq_encode.pt
--ckpt  ./checkpoint/EXE_GAN_model.pt  --masked_dir ./imgs/exe_guided_recovery/mask --gt_dir ./imgs/exe_guided_recovery/target --exemplar_dir ./imgs/exe_guided_recovery/exemplar --sample_times 2
> --eval_dir mixing_out  

```
- masked_dir: mask input folder
- gt_dir: the input gt_dir, used for  editing 
- exemplar_dir: exemplar_dir, the exemplar dir, for guiding the editing
- eval_dir: output dir
```
- Inputs are shown below:

| <img src="./imgs/exe_guided_recovery/style_mixing/1_real.png"  height=180 width=180 alt="Ground-truth"> | <img src="./imgs/exe_guided_recovery/style_mixing/1_mask.png" width=180 height=180 alt="Masked "> | <img src="./imgs/exe_guided_recovery/style_mixing/1_exe1.png" height=180 width=180 alt=" "> |<img src="./imgs/exe_guided_recovery/style_mixing/1_exe2.png" height=180 width=180 alt=" "> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |:----------------------------------------------------------: |
|         Ground-truth                      |                      Mask                              | Exemplar 1       | Exemplar 2  | 

- Style mixing results

| <img src="./imgs/exe_guided_recovery/mixing_out/1_0_0_inpaint2.png"  height=180 width=180 alt="Ground-truth"> | <img src="./imgs/exe_guided_recovery/mixing_out/1_1_0_inpaint2.png" width=180 height=180 alt="Masked "> | <img src="./imgs/exe_guided_recovery/mixing_out/1_2_0_inpaint2.png" height=180 width=180 alt=" "> |<img src="./imgs/exe_guided_recovery/mixing_out/1_3_0_inpaint2.png" height=180 width=180 alt=" "> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |:----------------------------------------------------------: |
| <img src="./imgs/exe_guided_recovery/mixing_out/1_4_0_inpaint2.png"  height=180 width=180 alt="Ground-truth"> | <img src="./imgs/exe_guided_recovery/mixing_out/1_5_0_inpaint2.png" width=180 height=180 alt="Masked "> | <img src="./imgs/exe_guided_recovery/mixing_out/1_6_0_inpaint2.png" height=180 width=180 alt=" "> |<img src="./imgs/exe_guided_recovery/mixing_out/1_7_0_inpaint2.png" height=180 width=180 alt=" "> |
| <img src="./imgs/exe_guided_recovery/mixing_out/1_0_0_inpaint.png"  height=180 width=180 alt="Ground-truth"> | <img src="./imgs/exe_guided_recovery/mixing_out/1_1_0_inpaint.png" width=180 height=180 alt="Masked "> | <img src="./imgs/exe_guided_recovery/mixing_out/1_2_0_inpaint.png" height=180 width=180 alt=" "> |<img src="./imgs/exe_guided_recovery/mixing_out/1_3_0_inpaint.png" height=180 width=180 alt=" "> |
| <img src="./imgs/exe_guided_recovery/mixing_out/1_4_0_inpaint.png"  height=180 width=180 alt="Ground-truth"> | <img src="./imgs/exe_guided_recovery/mixing_out/1_5_0_inpaint.png" width=180 height=180 alt="Masked "> | <img src="./imgs/exe_guided_recovery/mixing_out/1_6_0_inpaint.png" height=180 width=180 alt=" "> |<img src="./imgs/exe_guided_recovery/mixing_out/1_7_0_inpaint.png" height=180 width=180 alt=" "> |



## Bibtex
- If you find our code useful, please cite our paper:
  ```
  @misc{lu2022inpainting,
      title={Do Inpainting Yourself: Generative Facial Inpainting Guided by Exemplars}, 
      author={Wanglong Lu and Hanli Zhao and Xianta Jiang and Xiaogang Jin and Yongliang Yang and Min Wang and Jiankai Lyu and Kaijie Shi},
      year={2022},
      eprint={2202.06358},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
  }
  ```


## Acknowledgements

Model details and custom CUDA kernel codes are from official repositories: https://github.com/NVlabs/stylegan2

Codes for Learned Perceptual Image Patch Similarity, LPIPS came from https://github.com/richzhang/PerceptualSimilarity

To match FID scores more closely to tensorflow official implementations, I have used FID Inception V3 implementations in https://github.com/mseitzer/pytorch-fid
