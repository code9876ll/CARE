# CARE
### Prepare Datasets

Download the person datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), 
Then unzip them and rename them under the directory like

```
data
├── market1501
│   └── images ..
├── MSMT17
│   └── images ..
├── dukemtmcreid
    └── images ..

```

### Prepare ViT Pre-trained Models

You need to download the ImageNet pretrained transformer model : [ViT-Base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)
## Training

```bash
python train.py --config_file configs/transformer_base.yml MODEL.DEVICE_ID "('your device id')" MODEL.STRIDE_SIZE ${1} MODEL.SIE_CAMERA ${2} MODEL.SIE_VIEW ${3} MODEL.JPM ${4} MODEL.TRANSFORMER_TYPE ${5} OUTPUT_DIR ${OUTPUT_DIR} DATASETS.NAMES "('your dataset name')"
```
```bash
# using following commands:
sh dist_train.sh 
```

