require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

opt = {
    gpu = 1,
    init_g = '',
    init_d = '',
    init_r = '',

    txtSize = 1024,         -- #  of dim for raw text.
    nt = 128,               -- #  of dim for text features.
    nz = 10,               -- #  of dim for Z
    ngf = 64,              -- #  of gen filters in first conv layer
    ndf = 64,               -- #  of discrim filters in first conv layer
    fineSize = 64,
    loadSize = 76,
    batchSize = 128,
    numCaption = 4,
    nThreads = 4,           -- #  of data loading threads to use
    dataset = 'cub',       -- imagenet / lsun / folder
    save_every = 100,
    lr = 0.0002,            -- initial learning rate for adam
    lr_decay = 0.5,            -- initial learning rate for adam
    decay_every = 100,
    beta1 = 0.5,            -- momentum term of adam
    ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
    niter = 1000,             -- #  of iter at starting learning rate
    print_every = 4,
    nThreads = 4,
    replicate = 1,
    display = 1,            -- display samples while training. 0 = false
    display_id = 10,        -- display window id.
    name = 'experiment_long',

    data_root = '/home/xhy/code/textEditImage/dataset_cub/cub_icml',
    classnames = '/home/xhy/code/textEditImage/dataset_cub/cub_icml/allclasses.txt',
    trainids = '/home/xhy/code/textEditImage/dataset_cub/cub_icml/trainvalids.txt',
    img_dir = '/home/xhy/code/textEditImage/dataset_cub/CUB_200_2011/images',
    checkpoint_dir = '/home/xhy/code/textEditImage/checkpoints',

    netD_type = 'unet' -- unet
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

if opt.gpu > 0 then
   ok, cunn = pcall(require, 'cunn')
   ok2, cutorch = pcall(require, 'cutorch')
   cutorch.setDevice(opt.gpu)
end

if opt.display then disp = require 'display' end

local nt = opt.nt
local nz = opt.nz
local ngf = opt.ngf
local ndf = opt.ndf

--[[
opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
--]]

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
local Join = nn.JoinTable
local ReLU = nn.ReLU
local Dropout = nn.Dropout
local MaxPooling = nn.SpatialMaxPooling

local criterion_img = nn.MSECriterion()
local criterion = nn.BCECriterion()

local input_img_guide = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input_txt_guide = torch.Tensor(opt.batchSize, opt.txtSize)
local outputImage = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

local errD, errG, errW
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local function build_conv_block(dim, padding_type)
  local conv_block = nn.Sequential()
  local p = 0
  if padding_type == 'reflect' then
    conv_block:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  elseif padding_type == 'replicate' then
    conv_block:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
  elseif padding_type == 'zero' then
    p = 1
  end
  conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
  conv_block:add(SpatialBatchNormalization(dim))
  conv_block:add(nn.ReLU(true))
  if padding_type == 'reflect' then
    conv_block:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  elseif padding_type == 'replicate' then
    conv_block:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
  end
  conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
  conv_block:add(SpatialBatchNormalization(dim))
  return conv_block
end

local function build_res_block(dim, padding_type)
  local conv_block = build_conv_block(dim, padding_type)
  local res_block = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(conv_block)
  concat:add(nn.Identity())
  res_block:add(concat):add(nn.CAddTable())
  return res_block
end

local function ConvLayers(nIn, nOut, isSameSize, dropout)
  local kW, kH, dW, dH, padW, padH = 4, 4, 2, 2, 1, 1
  if isSameSize==1 then
    kW, kH, dW, dH, padW, padH = 3, 3, 1, 1, 1, 1 -- parameters for 'same' conv layers
  end

	local net = nn.Sequential()
	net:add(SpatialConvolution(nIn, nOut, kW, kH, dW, dH, padW, padH))
	net:add(SpatialBatchNormalization(nOut))
	net:add(ReLU(true))
	if dropout then net:add(Dropout(dropout)) end

	return net
end


if opt.init_g == '' and opt.netD_type == 'unet' then
    -- 输入图像+文本：图像64*3*64*64 文本64*1024
    local D1 = ConvLayers(3,64,0)()
    local D2 = ConvLayers(64,128,0)(D1)
    local D3 = ConvLayers(128,256,0)(D2)
    local D4 = ConvLayers(256,512,0)(D3)

    local encoder_txt = nn.Linear(opt.txtSize, opt.nt)()
    local encoder_txt_bn = SpatialBatchNormalization(opt.nt)(nn.View(-1,opt.nt,1,1)(encoder_txt))
    local encoder_txt_relu = nn.LeakyReLU(0.2,true)(nn.View(-1, opt.nt)(encoder_txt_bn))
    local encoder_txt_replicate = nn.Replicate(4,4)(nn.Replicate(4,3)(encoder_txt_relu))

    local syn = ConvLayers(512+opt.nt, ngf * 8, 1)(Join(2)({D4, encoder_txt_replicate}))

    local res1 = build_res_block(512, 'reflect')(syn)
    local res2 = build_res_block(512, 'reflect')(res1)
    local res3 = build_res_block(512, 'reflect')(res2)
    local res4 = build_res_block(512, 'reflect')(res3)
    local res5 = build_res_block(512, 'reflect')(res4)
    local res6 = build_res_block(512, 'reflect')(res5)
    local res7 = build_res_block(512, 'reflect')(res6)
    local res8 = build_res_block(512, 'reflect')(res7)
    local res9 = build_res_block(512, 'reflect')(res8)
    local res10 = build_res_block(512, 'reflect')(res9)
    local res11 = build_res_block(512, 'reflect')(res10)
    local res12 = build_res_block(512, 'reflect')(res11)

    local U4 = ConvLayers(1024, 512, 1)(Join(2)({res12, D4}))
    local U3 = ConvLayers(512, 256, 1)(Join(2)({ReLU(true)(SpatialFullConvolution(512,256, 4, 4, 2, 2, 1, 1)(U4)), D3}))
    local U2 = ConvLayers(256, 128, 1)(Join(2)({ReLU(true)(SpatialFullConvolution(256,128, 4, 4, 2, 2, 1, 1)(U3)), D2}))
    local U1 = ConvLayers(128, 64, 1)(Join(2)({ReLU(true)(SpatialFullConvolution(128, 64, 4, 4, 2, 2, 1, 1)(U2)), D1}))

    netG = nn.Sequential()
    netG:add(nn.gModule({D1,encoder_txt}, {U1}))
    netG:add(SpatialFullConvolution(64, 3, 4, 4, 2, 2, 1, 1))
    netG:add(nn.Tanh())

    netG:apply(weights_init)

elseif opt.init_g == '' then
    netG = nn.Sequential()

    encoder = nn.Sequential()
    -- 64*3*64*64
    encoder:add(SpatialConvolution(3, 64, 4, 4, 2, 2, 1, 1))
    encoder:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
    -- 64*64*32*32
    encoder:add(SpatialConvolution(64, 128, 4, 4, 2, 2, 1, 1))
    encoder:add(SpatialBatchNormalization(128)):add(nn.ReLU(true))
    -- 64*128*16*16
    encoder:add(SpatialConvolution(128, 256, 4, 4, 2, 2, 1, 1))
    encoder:add(SpatialBatchNormalization(256)):add(nn.ReLU(true))
    -- 64*256*8*8
    encoder:add(SpatialConvolution(256, 512, 4, 4, 2, 2, 1, 1))
    encoder:add(SpatialBatchNormalization(512)):add(nn.ReLU(true))
    -- 64*512*4*4

    local conc = nn.ConcatTable()
    local conv = nn.Sequential()
    conv:add(SpatialConvolution(ndf * 8, ndf * 4, 1, 1, 1, 1, 0, 0))
    conv:add(SpatialBatchNormalization(ndf * 4))
    conv:add(nn.LeakyReLU(0.2, true))
    conv:add(SpatialConvolution(ndf * 4, ndf * 4, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ndf * 4))
    conv:add(nn.LeakyReLU(0.2, true))
    conv:add(SpatialConvolution(ndf * 4, ndf * 8, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ndf * 8))
    conc:add(nn.Identity())
    conc:add(conv)
    encoder:add(conc)
    encoder:add(nn.CAddTable())
    encoder:add(nn.LeakyReLU(0.2, true))

    encoder_txt = nn.Sequential()
    encoder_txt:add(nn.Linear(opt.txtSize, opt.nt))
    encoder_txt:add(nn.BatchNormalization(opt.nt))
    encoder_txt:add(nn.LeakyReLU(0.2,true))
    encoder_txt:add(nn.Replicate(4,3))
    encoder_txt:add(nn.Replicate(4,4))

    ptg = nn.ParallelTable()
    ptg:add(encoder)
    ptg:add(encoder_txt)

    netG:add(ptg)
    netG:add(nn.JoinTable(2))
    netG:add(SpatialConvolution(512+opt.nt, ngf * 8, 3, 3, 1, 1, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 8))

    netG:add(build_res_block(ngf * 8, 'reflect')):add(build_res_block(ngf * 8, 'reflect')):add(build_res_block(ngf * 8, 'reflect')):add(build_res_block(ngf * 8, 'reflect'))
    netG:add(build_res_block(ngf * 8, 'reflect')):add(build_res_block(ngf * 8, 'reflect')):add(build_res_block(ngf * 8, 'reflect')):add(build_res_block(ngf * 8, 'reflect'))
    netG:add(build_res_block(ngf * 8, 'reflect')):add(build_res_block(ngf * 8, 'reflect')):add(build_res_block(ngf * 8, 'reflect')):add(build_res_block(ngf * 8, 'reflect'))
    netG:add(build_res_block(ngf * 8, 'reflect')):add(build_res_block(ngf * 8, 'reflect')):add(build_res_block(ngf * 8, 'reflect')):add(build_res_block(ngf * 8, 'reflect'))

    decoder = nn.Sequential()
    --64*512*4*4
    decoder:add(SpatialFullConvolution(512, 256, 4, 4, 2, 2, 1, 1))
    --decoder:add(nn.UpSampling(2, 'nearest'))
    --decoder:add(SpatialConvolution(512, 256, 3, 3, 1, 1, 1, 1))
    decoder:add(SpatialBatchNormalization(256)):add(nn.ReLU(true))
    --64*256*8*8
    decoder:add(SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1))
    --decoder:add(nn.UpSampling(2, 'nearest'))
    --decoder:add(SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1))
    decoder:add(SpatialBatchNormalization(128)):add(nn.ReLU(true))
    --64*128*16*16
    decoder:add(SpatialFullConvolution(128, 64, 4, 4, 2, 2, 1, 1))
    --decoder:add(nn.UpSampling(2, 'nearest'))
    --decoder:add(SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1))
    decoder:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
    --64*64*32*32
    decoder:add(SpatialFullConvolution(64, 3, 4, 4, 2, 2, 1, 1))
    --decoder:add(nn.UpSampling(2, 'nearest'))
    --decoder:add(SpatialConvolution(64, 3, 3, 3, 1, 1, 1, 1))
    decoder:add(nn.Tanh())
    --64*3*64*64

    netG:add(decoder)

    netG:apply(weights_init)
else
    netG = torch.load(opt.init_g)
end

if opt.gpu > 0 then
    input_img_guide = input_img_guide:cuda()
    input_txt_guide = input_txt_guide:cuda()
    outputImage = outputImage:cuda()
    netG = netG:cuda()
    criterion_img = criterion_img:cuda()
    criterion = criterion:cuda()
end

local parametersG, gradParametersG = netG:getParameters()

print(netG)
torch.save('./checkpoints_cub_reverseCycle_unet/netG.t7', netG)
-- netG = torch.load('./models/netG.t7')
-- print(netG)
