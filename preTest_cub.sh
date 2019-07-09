. CONFIG

net_gen=experiment_long_250_net_G.t7 \
net_gen_stage2=experiment_long_53_net_G_stage2.t7 \
queries=cub_queries.txt \
dataset=cub \
nThreads=1 \
net_txt=/home/xhy/code/textEditImage/dataset_cub/lm_sje_nc4_cub_hybrid_gru18_a1_c512_0.00070_1_10_trainvalids.txt_iter30000.t7 \
checkpoint_dir=/home/xhy/code/textEditImage/checkpoints_cub_reverseCycle \
stage2=0 \
th preTest.lua
