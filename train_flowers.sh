. CONFIG

init_d='/home/xhy/code/textEditImage/checkpoints_flowers_reverseCycle/experiment_long_253_net_D.t7' \
init_g='/home/xhy/code/textEditImage/checkpoints_flowers_reverseCycle/experiment_long_253_net_G.t7' \
init_g_2='/home/xhy/code/textEditImage/checkpoints_flowers_reverseCycle_stage2/experiment_long_39_net_G_stage2.t7' \
init_d_2='/home/xhy/code/textEditImage/checkpoints_flowers_reverseCycle_stage2/experiment_long_39_net_D_stage2.t7' \
dataset='flowers' \
net_txt='/home/xhy/code/textEditImage/dataset_flowers/lm_sje_flowers_c10_hybrid_0.00070_1_10_trainvalids.txt_iter16400.t7' \
data_root='/home/xhy/code/textEditImage/dataset_flowers/flowers_icml' \
classnames='/home/xhy/code/textEditImage/dataset_flowers/flowers_icml/allclasses.txt' \
trainids='/home/xhy/code/textEditImage/dataset_flowers/flowers_icml/trainvalids.txt' \
img_dir='/home/xhy/code/textEditImage/dataset_flowers/102flowers' \
checkpoint_dir='/home/xhy/code/textEditImage/checkpoints_flowers_reverseCycle' \
batchSize=128 \
epoch_begin=254 \
lambda1=10 \
lambda2=10 \
cycle_limit=4 \
cycle_limit_stage2=2.5 \
cls_weight=0.5 \
stage1=1 \
stage2=0 \
lr=0.00009 \
lr_stage2=0.00009 \
lr_decay=0.9 \
th train.lua