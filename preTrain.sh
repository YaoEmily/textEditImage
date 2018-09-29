. CONFIG

ID=1

display_id=10${ID} \
data_root=${CUB_META_DIR} \
classnames=${CUB_META_DIR}/allclasses.txt \
trainids=${CUB_META_DIR}/trainvalids.txt \
checkpoint_dir=${CHECKPOINT_DIR} \
th saveModels.lua
