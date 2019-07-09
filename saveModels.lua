require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

opt = {
    gpu = 1,
    init_g = '1',
    init_g_2 = '1',
    init_d = '1',
    init_d_2 = '1',
    init_g_v2 = '',
    init_d_v2_64 = '1',
    init_d_v2_128 = '1',

    txtSize = 1024,         -- #  of dim for raw text.
    nt = 128,               -- #  of dim for text features.
    nz = 10,               -- #  of dim for Z
    ngf = 64,              -- #  of gen filters in first conv layer
    ndf = 64,               -- #  of discrim filters in first conv layer
    fineSize = 64,
    loadSize = 76,
    batchSize = 64,
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

    netG_type = '' -- unet,
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

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
local Join = nn.JoinTable
local ReLU = nn.ReLU
local Dropout = nn.Dropout
local MaxPooling = nn.SpatialMaxPooling

local criterion_img = nn.MSECriterion()
local criterion = nn.BCECriterion()

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

local function FullConvLayers(nIn, nOut, dropout)
  local kW, kH, dW, dH, padW, padH = 4, 4, 2, 2, 1, 1
  local net = nn.Sequential()
  net:add(SpatialFullConvolution(nIn, nOut, kW, kH, dW, dH, padW, padH))
  net:add(SpatialBatchNormalization(nOut))
  net:add(ReLU(true))

  if dropout then net:add(Dropout(dropout)) end

  return net
end

if opt.init_g == '' and opt.netG_type == 'unet' then
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
    local U3 = SpatialFullConvolution(512,256, 4, 4, 2, 2, 1, 1)(U4)
    local U2 = SpatialFullConvolution(256,128, 4, 4, 2, 2, 1, 1)(U3)
    local U1 = SpatialFullConvolution(128, 64, 4, 4, 2, 2, 1, 1)(U2)

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
end

if opt.init_g_2 == '' then
    netG_stage2 = nn.Sequential()

    encoder_img = nn.Sequential()

    encoder_real = nn.Sequential()
    -- 64*3*128*128
    encoder_real:add(SpatialConvolution(3, 64, 4, 4, 2, 2, 1, 1))
    encoder_real:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
    -- 64*64*64*64
    encoder_real:add(SpatialConvolution(64, 128, 4, 4, 2, 2, 1, 1))
    encoder_real:add(SpatialBatchNormalization(128)):add(nn.ReLU(true))
    -- 64*128*32*32
    encoder_real:add(SpatialConvolution(128, 256, 4, 4, 2, 2, 1, 1))
    encoder_real:add(SpatialBatchNormalization(256)):add(nn.ReLU(true))
    -- 64*256*16*16
    encoder_real:add(SpatialConvolution(256, 512, 4, 4, 2, 2, 1, 1))
    encoder_real:add(SpatialBatchNormalization(512)):add(nn.ReLU(true))
    -- 64*512*8*8
    encoder_real:add(SpatialConvolution(512, 1024, 4, 4, 2, 2, 1, 1))
    encoder_real:add(SpatialBatchNormalization(1024)):add(nn.ReLU(true))
    -- 64*1024*4*4

    encoder_stage1 = nn.Sequential()
    -- 64*3*64*64
    encoder_stage1:add(SpatialConvolution(3, 64, 4, 4, 2, 2, 1, 1))
    encoder_stage1:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
    -- 64*64*32*32
    encoder_stage1:add(SpatialConvolution(64, 128, 4, 4, 2, 2, 1, 1))
    encoder_stage1:add(SpatialBatchNormalization(128)):add(nn.ReLU(true))
    -- 64*128*16*16
    encoder_stage1:add(SpatialConvolution(128, 256, 4, 4, 2, 2, 1, 1))
    encoder_stage1:add(SpatialBatchNormalization(256)):add(nn.ReLU(true))
    -- 64*256*8*8
    encoder_stage1:add(SpatialConvolution(256, 512, 4, 4, 2, 2, 1, 1))
    encoder_stage1:add(SpatialBatchNormalization(512)):add(nn.ReLU(true))
    -- 64*512*4*4

    ptg1 = nn.ParallelTable()
    ptg1:add(encoder_real)
    ptg1:add(encoder_stage1)

    encoder_img:add(ptg1)
    encoder_img:add(nn.JoinTable(2))
    encoder_img:add(SpatialConvolution(1024+512, ngf * 16, 3, 3, 1, 1, 1, 1))
    encoder_img:add(SpatialBatchNormalization(ngf * 16))
    -- 64*1024*4*4

    encoder_txt = nn.Sequential()
    encoder_txt:add(nn.Linear(opt.txtSize, opt.nt))
    encoder_txt:add(nn.BatchNormalization(opt.nt))
    encoder_txt:add(nn.LeakyReLU(0.2,true))
    encoder_txt:add(nn.Replicate(4,3))
    encoder_txt:add(nn.Replicate(4,4))
    -- 64*256*4*4

    ptg2 = nn.ParallelTable()
    ptg2:add(encoder_img)
    ptg2:add(encoder_txt)

    netG_stage2:add(ptg2)
    netG_stage2:add(nn.JoinTable(2))
    netG_stage2:add(SpatialConvolution(1024+256, ngf * 16, 3, 3, 1, 1, 1, 1))
    netG_stage2:add(SpatialBatchNormalization(ngf * 16))

    netG_stage2:add(build_res_block(ngf * 16, 'reflect')):add(build_res_block(ngf * 16, 'reflect'))
    netG_stage2:add(build_res_block(ngf * 16, 'reflect')):add(build_res_block(ngf * 16, 'reflect'))

    decoder = nn.Sequential()
    --64*1024*4*4
    decoder:add(SpatialFullConvolution(1024, 512, 4, 4, 2, 2, 1, 1))
    decoder:add(SpatialBatchNormalization(512)):add(nn.ReLU(true))
    --64*512*8*8
    decoder:add(SpatialFullConvolution(512, 256, 4, 4, 2, 2, 1, 1))
    decoder:add(SpatialBatchNormalization(256)):add(nn.ReLU(true))
    --64*256*16*16
    decoder:add(SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1))
    decoder:add(SpatialBatchNormalization(128)):add(nn.ReLU(true))
    --64*128*32*32
    decoder:add(SpatialFullConvolution(128, 64, 4, 4, 2, 2, 1, 1))
    decoder:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
    --64*64*64*64
    decoder:add(SpatialFullConvolution(64, 3, 4, 4, 2, 2, 1, 1))
    decoder:add(nn.Tanh())
    --64*3*128*128

    netG_stage2:add(decoder)

    netG_stage2:apply(weights_init)
end

if opt.init_d_2 == '' then
    convD = nn.Sequential()
    -- input is 3 x 128 x 128
    convD:add(SpatialConvolution(3, 64, 4, 4, 2, 2, 1, 1))
    convD:add(nn.LeakyReLU(0.2, true))
    -- state size: 64 x 64 x 64
    convD:add(SpatialConvolution(64, 128, 4, 4, 2, 2, 1, 1))
    convD:add(SpatialBatchNormalization(128)):add(nn.LeakyReLU(0.2, true))
    -- state size: 128 x 32 x 32
    convD:add(SpatialConvolution(128, 256, 4, 4, 2, 2, 1, 1))
    convD:add(SpatialBatchNormalization(256))
    -- state size: 256 x 16 x 16
    convD:add(SpatialConvolution(256, 512, 4, 4, 2, 2, 1, 1))
    convD:add(SpatialBatchNormalization(512))
    -- state size: 512 x 8 x 8
    convD:add(SpatialConvolution(512, 1024, 4, 4, 2, 2, 1, 1))
    convD:add(SpatialBatchNormalization(1024))
    -- state size: 1024 x 4 x 4

    -- state size: 1024 x 4 x 4
    local conc = nn.ConcatTable()
    local conv = nn.Sequential()
    conv:add(SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0))
    conv:add(SpatialBatchNormalization(256)):add(nn.LeakyReLU(0.2, true))
    conv:add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(256))
    conv:add(nn.LeakyReLU(0.2, true))
    conv:add(SpatialConvolution(256, 1024, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(1024))
    conc:add(nn.Identity())
    conc:add(conv)
    convD:add(conc)
    convD:add(nn.CAddTable())
    convD:add(nn.LeakyReLU(0.2, true))

    local fcD = nn.Sequential()
    fcD:add(nn.Linear(opt.txtSize,opt.nt))
    fcD:add(nn.BatchNormalization(opt.nt))
    fcD:add(nn.LeakyReLU(0.2,true))
    fcD:add(nn.Replicate(4,3))
    fcD:add(nn.Replicate(4,4))

    netD_stage2 = nn.Sequential()

    pt = nn.ParallelTable()
    pt:add(convD)
    pt:add(fcD)

    netD_stage2:add(pt)
    netD_stage2:add(nn.JoinTable(2))

    -- state size: (1024 + opt.nt) x 4 x 4
    netD_stage2:add(SpatialConvolution(1024 + opt.nt, 1024, 1, 1))
    netD_stage2:add(SpatialBatchNormalization(1024)):add(nn.LeakyReLU(0.2, true))
    netD_stage2:add(SpatialConvolution(1024, 1, 4, 4))
    netD_stage2:add(nn.Sigmoid())

    -- state size: 1 x 1 x 1
    netD_stage2:add(nn.View(1):setNumInputDims(3))
    -- state size: 1

    netD_stage2:apply(weights_init)
end

if opt.init_g_v2 == '' then

    -- 输入图像+文本：图像64*3*64*64 文本64*1024
    local input = nn.Identity()()
    local pool_64 = MaxPooling(2, 2, 2, 2)(input) --3*64*64
    local D1_64 = ConvLayers(3,64,0)(pool_64) --64*32*32
    local D2_64 = ConvLayers(64,128,0)(D1_64) --128*16*16
    local D3_64 = ConvLayers(128,256,0)(D2_64) --256*8*8
    local D4_64 = ConvLayers(256,512,0)(D3_64) --512*4*4

    local encoder_txt = nn.Linear(opt.txtSize, opt.nt)()
    local encoder_txt_bn = SpatialBatchNormalization(opt.nt)(nn.View(-1,opt.nt,1,1)(encoder_txt))
    local encoder_txt_relu = nn.LeakyReLU(0.2,true)(nn.View(-1, opt.nt)(encoder_txt_bn))
    local encoder_txt_replicate_4 = nn.Replicate(4,4)(nn.Replicate(4,3)(encoder_txt_relu))

    local syn = ConvLayers(512+opt.nt, ngf * 8, 1)(Join(2)({D4_64, encoder_txt_replicate_4}))

    local res1_64 = build_res_block(512, 'reflect')(syn)
    local res2_64 = build_res_block(512, 'reflect')(res1_64)
    local res3_64 = build_res_block(512, 'reflect')(res2_64)
    local res4_64 = build_res_block(512, 'reflect')(res3_64)
    local res5_64 = build_res_block(512, 'reflect')(res4_64)
    local res6_64 = build_res_block(512, 'reflect')(res5_64)
    local res7_64 = build_res_block(512, 'reflect')(res6_64)
    local res8_64 = build_res_block(512, 'reflect')(res7_64)
    local res9_64 = build_res_block(512, 'reflect')(res8_64)
    local res10_64 = build_res_block(512, 'reflect')(res9_64)
    local res11_64 = build_res_block(512, 'reflect')(res10_64)
    local res12_64 = build_res_block(512, 'reflect')(res11_64)
    local res13_64 = build_res_block(512, 'reflect')(res12_64)
    local res14_64 = build_res_block(512, 'reflect')(res13_64)
    local res15_64 = build_res_block(512, 'reflect')(res14_64)
    local res16_64 = build_res_block(512, 'reflect')(res15_64)
    local res17_64 = build_res_block(512, 'reflect')(res16_64)
    local res18_64 = build_res_block(512, 'reflect')(res17_64)
    local res19_64 = build_res_block(512, 'reflect')(res18_64)
    local res20_64 = build_res_block(512, 'reflect')(res19_64) --512*4*4

    local U4_64 = FullConvLayers(512, 256)(res20_64) --256*8*8
    local U3_64 = FullConvLayers(256, 128)(U4_64) --128*16*16
    local U2_64 = FullConvLayers(128, 64)(U3_64) --256*32*32
    local U1_64 = FullConvLayers(64, 32)(U2_64) -- 32*64*64

    local img_64 = nn.Tanh()(SpatialConvolution(32, 3, 3, 3, 1, 1, 1, 1)(U1_64))

    local D1_128 = ConvLayers(3,32,0)(input) --32*64*64
    local syn1 = ConvLayers(32 + 32, 64, 1)(Join(2)({D1_128, U1_64}))

    local encoder_txt_replicate_64 = nn.Replicate(64,4)(nn.Replicate(64,3)(encoder_txt_relu)) --opt.nt*64*64
    local syn2 = ConvLayers(64 + 128, 128, 1)(Join(2)({syn1, encoder_txt_replicate_64}))

    local res1_4_128 = build_res_block(128, 'reflect')(build_res_block(128, 'reflect')(build_res_block(128, 'reflect')(build_res_block(128, 'reflect')(syn2))))
    local res5_8_128 = build_res_block(128, 'reflect')(build_res_block(128, 'reflect')(build_res_block(128, 'reflect')(build_res_block(128, 'reflect')(res1_4_128))))

    local U3_128 = FullConvLayers(128, 64)(res5_8_128) --64*128*128
    local U2_128 = ConvLayers(64, 32, 1)(U3_128) -- 32*128*128
    local U1_128 = ConvLayers(32, 16, 1)(U2_128) -- 16*128*128

    local img_128 = nn.Tanh()(SpatialConvolution(16, 3, 3, 3, 1, 1, 1, 1)(U1_128))

    netG_v2 = nn.Sequential()
    netG_v2:add(nn.gModule({input, encoder_txt}, {img_64, img_128}))

    netG_v2:apply(weights_init)
end

if opt.init_d_v2_64 == '' then
    local D1_64 = ConvLayers(3,64,0)()
    local D2_64 = ConvLayers(64,128,0)(D1_64)
    local D3_64 = ConvLayers(128,256,0)(D2_64)
    local D4_64 = ConvLayers(256,512,0)(D3_64) --512*8*8

    local B1_1 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(256)(SpatialConvolution(512, 256, 1, 1)(D4_64)))
    local B1_2 = nn.View(1):setNumInputDims(3)(nn.Sigmoid()(SpatialConvolution(256, 1, 4, 4)(B1_1)))

    local encoder_txt = nn.Linear(opt.txtSize, opt.nt)()
    local encoder_txt_bn = SpatialBatchNormalization(opt.nt)(nn.View(-1,opt.nt,1,1)(encoder_txt))
    local encoder_txt_relu = nn.LeakyReLU(0.2,true)(nn.View(-1, opt.nt)(encoder_txt_bn))
    local encoder_txt_replicate = nn.Replicate(8,4)(nn.Replicate(8,3)(encoder_txt_relu)) --opt.nt*8*8

    local B2_1 = Join(2)({D4_64, encoder_txt_replicate})
    local B2_2 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(512)(SpatialConvolution(512 + opt.nt, 512, 1, 1)(B2_1)))
    local B2_3 = nn.Sigmoid()(SpatialConvolution(512, 1, 4, 4)(B2_2))
    local B2_4 = nn.View(1):setNumInputDims(3)(B2_3)

    netD_v2_64 = nn.Sequential()
    netD_v2_64:add(nn.gModule({D1_64, encoder_txt}, {B1_2, B2_4}))

    netD_v2_64:apply(weights_init)
end

if opt.init_d_v2_128 == '' then
    local D1_128 = ConvLayers(3,64,0)()
    local D2_128 = ConvLayers(64,128,0)(D1_128)
    local D3_128 = ConvLayers(128,256,0)(D2_128)
    local D4_128 = ConvLayers(256,512,0)(D3_128) --512*8*8

    local B1_1 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(256)(SpatialConvolution(512, 256, 1, 1)(D4_128))) --256*8*8
    local B1_2 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(256)(SpatialConvolution(256, 256, 4, 4, 2, 2, 1, 1)(B1_1))) --256*4*4
    local B1_3 = nn.View(1):setNumInputDims(3)(nn.Sigmoid()(SpatialConvolution(256, 1, 4, 4)(B1_2)))

    local encoder_txt = nn.Linear(opt.txtSize, opt.nt)()
    local encoder_txt_bn = SpatialBatchNormalization(opt.nt)(nn.View(-1,opt.nt,1,1)(encoder_txt))
    local encoder_txt_relu = nn.LeakyReLU(0.2,true)(nn.View(-1, opt.nt)(encoder_txt_bn))
    local encoder_txt_replicate = nn.Replicate(8,4)(nn.Replicate(8,3)(encoder_txt_relu))

    local B2_1 = Join(2)({D4_128, encoder_txt_replicate})
    local B2_2 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(512)(SpatialConvolution(512 + opt.nt, 512, 1, 1)(B2_1))) --512*8*8
    local B2_3 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(256)(SpatialConvolution(512, 256, 4, 4, 2, 2, 1, 1)(B2_2))) --256*4*4
    local B2_4 = nn.Sigmoid()(SpatialConvolution(256, 1, 4, 4)(B2_3))
    local B2_5 = nn.View(1):setNumInputDims(3)(B2_4)

    netD_v2_128 = nn.Sequential()
    netD_v2_128:add(nn.gModule({D1_128, encoder_txt}, {B1_3, B2_5}))

    netD_v2_128:apply(weights_init)
end

--print(netG)
--torch.save('./checkpoints_cub_reverseCycle_resBlock24/netG.t7', netG)
--print(netG_stage2)
--torch.save('./checkpoints_flowers_reverseCycle_stage2/netG_stage2.t7', netG_stage2)
--print(netD_stage2)
--torch.save('./checkpoints_cub_reverseCycle_stage2/netD_stage2.t7', netD_stage2)

print(netG_v2)
torch.save('./checkpoints_flowers_reverseCycle_v2/netG_v2.t7', netG_v2)
--print(netD_v2_64)
--torch.save('./checkpoints_flowers_reverseCycle_v2/netD_v2_64.t7', netD_v2_64)

--print(netD_v2_128)
--torch.save('./checkpoints_flowers_reverseCycle_v2/netD_v2_128.t7', netD_v2_128)
