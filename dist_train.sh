# train
CUDA_VISIBLE_DEVICES=4,5 python train.py --config_file configs/DukeMTMC/vit_base.yml
# test
#python test.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"
