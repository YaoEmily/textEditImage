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
local SpatialFullConvolution = nn.SpatialFullConvolution --去卷积或上采样的操作

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

if opt.init_g == '' then
    -- 输入图像+文本：图像64*3*64*64 文本64*1024

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
torch.save('./checkpoints_cub_reverseCycle/netG.t7', netG)
-- netG = torch.load('./models/netG.t7');
-- print(netG)

--[[
local preNetG = function(x)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    gradParametersG:zero()

    data_tm:reset(); data_tm:resume()
    real_img, real_txt, wrong_img, _ = data:getBatch()
    data_tm:stop()

    input_img_guide:copy(real_img)
    input_txt_guide:copy(real_txt)
    --print(input_img_guide:size()) --64*3*64*64
    --print(input_txt_guide:size()) --64*1024
    local fake = netG:forward(input_img_guide)
    outputImage:copy(fake)

    errG = criterion_img:forward(input_img_guide, outputImage)
    local gradient = criterion_img:backward(input_img_guide, outputImage)
    netG:backward(input_img_guide, gradient)
    --netG:updateParameters(opt.lr)
    return errG, gradParametersG
end

-- train
for epoch = 1, opt.niter do
    epoch_tm:reset()

    if epoch % opt.decay_every == 0 then
        optimStateG.learningRate = optimStateG.learningRate * opt.lr_decay
        --optimStateD.learningRate = optimStateD.learningRate * opt.lr_decay
    end

    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        tm:reset()

        -- (2) Update G network: maximize log(D(G(z)))
        optim.adam(preNetG, parametersG, optimStateG)
        -- logging
        if ((i-1) / opt.batchSize) % opt.print_every == 0 then
            print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                .. '  Err_G: %.4f'):format(
                epoch,
                ((i-1) / opt.batchSize),
                math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                tm:time().real,
                data_tm:time().real,
                optimStateG.learningRate,
                errG and errG or -1))
            local fake = netG.output
            --print(fake:size()) -- 64*3*64*64
            disp.image(fake:narrow(1,1,opt.batchSize), {win=opt.display_id, title=opt.name})
            disp.image(real_img, {win=opt.display_id * 3, title=opt.name})
        end
    end

    -- save checkpoints
    if epoch % opt.save_every == 0 then
      paths.mkdir(opt.checkpoint_dir)
      torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG)
      --torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD)
      --torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_opt.t7', opt)
      print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
          epoch, opt.niter, epoch_tm:time().real))
      end
end
--]]
