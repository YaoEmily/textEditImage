. CONFIG

net_gen=experiment_long_41_net_G.t7 \
queries=flowers_queries.txt \
dataset=flowers \
nThreads=1 \
net_txt='/home/xhy/code/textEditImage/dataset_flowers/lm_sje_flowers_c10_hybrid_0.00070_1_10_trainvalids.txt_iter16400.t7' \
data_root='/home/xhy/code/textEditImage/dataset_flowers/flowers_icml' \
classnames='/home/xhy/code/textEditImage/dataset_flowers/flowers_icml/allclasses.txt' \
trainids='/home/xhy/code/textEditImage/dataset_flowers/flowers_icml/trainvalids.txt' \
img_dir='/home/xhy/code/textEditImage/dataset_flowers/102flowers' \
checkpoint_dir='/home/xhy/code/textEditImage/checkpoints_flowers_reverseCycle_v2' \
th preTest_v2.lua
