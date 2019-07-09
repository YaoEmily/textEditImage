require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'dpnn'
require 'image'

opt = {
   numCaption = 4,
   replicate = 1, -- if 1, then replicate averaged text features numCaption times.
   save_every = 1,
   print_every = 1,
   dataset = 'cub',       -- imagenet / lsun / folder
   no_aug = 0,
   keep_img_frac = 1.0,
   cls_weight = 0.5,
   filenames = '',
   data_root = '/home/xhy/code/textEditImage/dataset_cub/cub_icml',
   classnames = '/home/xhy/code/textEditImage/dataset_cub/cub_icml/allclasses.txt',
   trainids = '/home/xhy/code/textEditImage/dataset_cub/cub_icml/trainvalids.txt',
   img_dir = '/home/xhy/code/textEditImage/dataset_cub/CUB_200_2011/images',
   checkpoint_dir = '/home/xhy/code/textEditImage/checkpoints_cub_reverseCycle',
   numshot = 0,
   batchSize = 64,
   doc_length = 201,
   loadSize = 76,
   loadSize_stage2 = 140,
   txtSize = 1024,         -- #  of dim for raw text.
   fineSize = 64,
   fineSize_stage2 = 128,
   nt = 128,               -- #  of dim for text features.
   nz = 10,               -- #  of dim for Z
   ngf = 64,              -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 1000,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   lr_decay = 0.5,            -- initial learning rate for adam
   decay_every = 100,
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment_long',
   noise = 'normal',       -- uniform / normal
   use_cudnn = 0,

   init_g = '/home/xhy/code/textEditImage/checkpoints_cub2/netG.t7',
   init_d_64 = '',
   init_d_128 = '',
   net_txt = '/home/xhy/code/textEditImage/dataset_cub/lm_sje_nc4_cub_hybrid_gru18_a1_c512_0.00070_1_10_trainvalids.txt_iter30000.t7',

   lambda1 = 2,
   lambda2 = 2,
   lambda_identity = 0.5,
   epoch_begin = 1,
   cycle_limit = 2,
   cycle_limit_stage2 = 3,
   stage1 = 1,
   stage2 = 0,
   train = 1,
   lr_stage2 = 0.0002,
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

if opt.display == 0 then opt.display = false end

if opt.gpu > 0 then
   ok, cunn = pcall(require, 'cunn')
   require 'cudnn'
   ok2, cutorch = pcall(require, 'cutorch')
   cutorch.setDevice(opt.gpu)
end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

net_txt = torch.load(opt.net_txt)
if net_txt.protos ~=nil then net_txt = net_txt.protos.enc_doc end
net_txt:evaluate()

----------------------------------------------------------------------------
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

lambda1 = opt.lambda1
lambda2 = opt.lambda2

local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

netG = torch.load(opt.init_g)
netD_64 = torch.load(opt.init_d_64)
netD_128 = torch.load(opt.init_d_128)

--netG:apply(weights_init)
--netD:apply(weights_init)
--print(netG)
--print(netD)
--print(net_txt)

local criterion = nn.BCECriterion()
local absCriterion = nn.AbsCriterion()
local mseCriterion = nn.MSECriterion()

optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD_64 = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD_128 = {
   learningRate = opt.lr_stage2,
   beta1 = opt.beta1,
}

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
alphabet_size = #alphabet
local input_img_real = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input_img_wrong = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input_img_real_128 = torch.Tensor(opt.batchSize, 3, opt.fineSize_stage2, opt.fineSize_stage2)
local input_img_wrong_128 = torch.Tensor(opt.batchSize, 3, opt.fineSize_stage2, opt.fineSize_stage2)
if opt.replicate == 1 then
  input_txt_real_raw = torch.Tensor(opt.batchSize, opt.doc_length, #alphabet)
  input_txt_wrong_raw = torch.Tensor(opt.batchSize, opt.doc_length, #alphabet)
else
  input_txt_real_raw = torch.Tensor(opt.batchSize * opt.numCaption, opt.txtSize)
  input_txt_wrong_raw = torch.Tensor(opt.batchSize * opt.numCaption, opt.txtSize)
end
local input_txt_real = torch.Tensor(opt.batchSize, opt.txtSize)
local input_txt_wrong = torch.Tensor(opt.batchSize, opt.txtSize)
local label = torch.Tensor(opt.batchSize)
local errDA, errGA, errRA, errWA, errFA, errIA, errRecA, errAdaptA
local errDB, errGB, errRB, errWB, errFB, errIB, errRecB, errAdaptB
local errDA_stage2, errGA_stage2, errRA_stage2, errWA_stage2, errFA_stage2, errIA_stage2, errRecA_stage2, errAdaptA_stage2
local errDB_stage2, errGB_stage2, errRB_stage2, errWB_stage2, errFB_stage2, errIB_stage2, errRecB_stage2, errAdaptB_stage2
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

if opt.gpu > 0 then
    input_img_real = input_img_real:cuda()
    input_img_wrong = input_img_wrong:cuda()
    input_img_real_128 = input_img_real_128:cuda()
    input_img_wrong_128 = input_img_wrong_128:cuda()
    input_txt_real = input_txt_real:cuda()
    input_txt_wrong = input_txt_wrong:cuda()
    input_txt_real_raw = input_txt_real_raw:cuda()
    input_txt_wrong_raw = input_txt_wrong_raw:cuda()
    label = label:cuda()
    netD_64:cuda()
    netD_128:cuda()
    netG:cuda()
    net_txt:cuda()
    criterion:cuda()
    absCriterion:cuda()
    mseCriterion:cuda()
end

if opt.use_cudnn == 1 then
  cudnn = require('cudnn')
  netD_64 = cudnn.convert(netD_64, cudnn)
  netD_128 = cudnn.convert(netD_128, cudnn)
  netG = cudnn.convert(netG, cudnn)
  net_txt = cudnn.convert(net_txt, cudnn)
end

local parametersD_64, gradParametersD_64 = netD_64:getParameters()
local parametersD_128, gradParametersD_128 = netD_128:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

local fDxA_64 = function(x)
  netD_64:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersD_64:zero()

  -- train with real
  label:fill(real_label)
  local output = netD_64:forward{input_img_real, input_txt_real}
  local errD_real_uncond = criterion:forward(output[1], label)
  local df_do_uncond = criterion:backward(output[1], label)
  local df_do_uncond_clone = df_do_uncond:clone()
  local errD_real = criterion:forward(output[2], label)
  local df_do = criterion:backward(output[2], label)
  local df_do_clone = df_do:clone()
  netD_64:backward({input_img_real, input_txt_real}, {df_do_uncond_clone, df_do_clone})

  -- train with wrong
  label:fill(real_label)
  local output = netD_64:forward({input_img_wrong, input_txt_real})
  local errD_wrong_uncond = opt.cls_weight * criterion:forward(output[1], label)
  local df_do_uncond = criterion:backward(output[1], label):mul(opt.cls_weight)
  local df_do_uncond_clone = df_do_uncond:clone()
  label:fill(fake_label)
  local errD_wrong = opt.cls_weight * criterion:forward(output[2], label)
  local df_do = criterion:backward(output[2], label):mul(opt.cls_weight)
  local df_do_clone = df_do:clone()
  netD_64:backward({input_img_wrong, input_txt_real}, {df_do_uncond_clone, df_do_clone})

  -- train with fake
  local fake_weight = 1 - opt.cls_weight
  local fake = netG:forward({input_img_wrong_128, input_txt_real})
  local output = netD_64:forward{fake[1], input_txt_real}
  label:fill(fake_label)
  local errD_fake_uncond = fake_weight * criterion:forward(output[1], label)
  local df_do_uncond = criterion:backward(output[1], label):mul(fake_weight)
  local df_do_uncond_clone = df_do_uncond:clone()
  local errD_fake = fake_weight * criterion:forward(output[2], label)
  local df_do = criterion:backward(output[2], label):mul(fake_weight)
  local df_do_clone = df_do:clone()
  netD_64:backward({fake[1], input_txt_real}, {df_do_uncond_clone, df_do_clone})

  errRA_64 = errD_real + errD_real_uncond
  errWA_64 = errD_wrong + errD_wrong_uncond
  errFA_64 = errD_fake + errD_fake_uncond
  errDA_64 = errRA_64 + errWA_64 + errFA_64

  return errDA_64, gradParametersD_64
end


local fDxA_128 = function(x)
  netD_128:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersD_64:zero()

  -- train with real
  label:fill(real_label)
  local output = netD_128:forward{input_img_real_128, input_txt_real}
  local errD_real_uncond = criterion:forward(output[1], label)
  local df_do_uncond = criterion:backward(output[1], label)
  local df_do_uncond_clone = df_do_uncond:clone()
  local errD_real = criterion:forward(output[2], label)
  local df_do = criterion:backward(output[2], label)
  local df_do_clone = df_do:clone()
  netD_128:backward({input_img_real_128, input_txt_real}, {df_do_uncond_clone, df_do_clone})

  -- train with wrong
  errD_wrong = 0
  label:fill(real_label)
  local output = netD_128:forward({input_img_wrong_128, input_txt_real})
  local errD_wrong_uncond = opt.cls_weight * criterion:forward(output[1], label)
  local df_do_uncond = criterion:backward(output[1], label):mul(opt.cls_weight)
  local df_do_uncond_clone = df_do_uncond:clone()
  label:fill(fake_label)
  local errD_wrong = opt.cls_weight * criterion:forward(output[2], label)
  local df_do = criterion:backward(output[2], label):mul(opt.cls_weight)
  local df_do_clone = df_do:clone()
  netD_128:backward({input_img_wrong_128, input_txt_real}, {df_do_uncond_clone, df_do_clone})

  -- train with fake
  local fake_weight = 1 - opt.cls_weight
  local fake = netG:forward({input_img_wrong_128, input_txt_real})
  local output = netD_128:forward{fake[2], input_txt_real}
  label:fill(fake_label)
  local errD_fake_uncond = fake_weight * criterion:forward(output[1], label)
  local df_do_uncond = criterion:backward(output[1], label):mul(fake_weight)
  local df_do_uncond_clone = df_do_uncond:clone()
  local errD_fake = fake_weight * criterion:forward(output[2], label)
  local df_do = criterion:backward(output[2], label):mul(fake_weight)
  local df_do_clone = df_do:clone()
  netD_128:backward({fake[2], input_txt_real}, {df_do_uncond_clone, df_do_clone})

  errRA_128 = errD_real + errD_real_uncond
  errWA_128 = errD_wrong + errD_wrong_uncond
  errFA_128 = errD_fake + errD_fake_uncond
  errDA_128 = errRA_128 + errWA_128 + errFA_128

  return errDA_128, gradParametersD_128
end

local x64 = torch.rand(opt.batchSize, 3, 64, 64):cuda()
local x128 = torch.rand(opt.batchSize, 3, 128, 128):cuda()

-- create closure to evaluate f(X) and df/dX of generator
local fGxA = function(x)
  netD_64:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netD_128:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersG:zero()

  -- G should be identity if corresponding image and text pair is fed
  local identity = nil
  if opt.lambda_identity > 0 then
    identity = netG:forward({input_img_real_128, input_txt_real})
    local errI_64 = absCriterion:forward(identity[1], input_img_real) * lambda2 * opt.lambda_identity
    local didentity_loss_do_64 = absCriterion:backward(identity[1], input_img_real):mul(lambda2):mul(opt.lambda_identity)
    local didentity_loss_do_64_clone = didentity_loss_do_64:clone()
    local errI_128 = absCriterion:forward(identity[2], input_img_real_128) * lambda2 * opt.lambda_identity
    local didentity_loss_do_128 = absCriterion:backward(identity[2], input_img_real_128):mul(lambda2):mul(opt.lambda_identity)
    netG:backward({input_img_real_128, input_txt_real}, {didentity_loss_do_64_clone, didentity_loss_do_128})
    errIA = errI_64 + errI_128
  end

  -- GAN loss
  label:fill(real_label)
  local fake = netG:forward({input_img_wrong_128, input_txt_real})
  local output_64 = netD_64:forward({fake[1], input_txt_real})
  local errGA_64_uncond = criterion:forward(output_64[1], label)
  local df_do_64_uncond = criterion:backward(output_64[1], label)
  local df_do_64_uncond_clone = df_do_64_uncond:clone()
  local errGA_64 = criterion:forward(output_64[2], label)
  local df_do_64 = criterion:backward(output_64[2], label)
  local df_do_64_clone = df_do_64:clone()

  local output_128 = netD_128:forward({fake[2], input_txt_real})
  local errGA_128_uncond = criterion:forward(output_128[1], label)
  local df_do_128_uncond = criterion:backward(output_128[1], label)
  local df_do_128_uncond_clone = df_do_128_uncond:clone()
  local errGA_128 = criterion:forward(output_128[2], label)
  local df_do_128 = criterion:backward(output_128[2], label)
  local df_do_128_clone = df_do_128:clone()

  local df_d_GAN_64 = netD_64:updateGradInput({fake[1], input_txt_real}, {df_do_64_uncond_clone, df_do_64_clone})
  local df_d_GAN_64_clone = df_d_GAN_64[1]:clone()
  local df_d_GAN_128 = netD_128:updateGradInput({fake[2], input_txt_real}, {df_do_128_uncond_clone, df_do_128_clone})
  local df_d_GAN_128_clone = df_d_GAN_128[1]:clone()

  errGA = errGA_64_uncond + errGA_64 + errGA_128_uncond + errGA_128

  errRecA = -1
  errAdaptA = -1

  if errGA > opt.cycle_limit then
    netG:backward({input_img_wrong_128, input_txt_real}, {df_d_GAN_64_clone, df_d_GAN_128_clone})
    print("no cycle loss for GA")
    return errGA, gradParametersG
  end

  -- forward cycle loss
  local rec = netG:forward({fake[2], input_txt_wrong})
  local errRecA_64 = absCriterion:forward(rec[1], input_img_wrong) * opt.lambda1
  local df_do_cycle_64 = absCriterion:backward(rec[1], input_img_wrong):mul(opt.lambda1)
  local df_do_cycle_64_clone = df_do_cycle_64:clone()
  local errRecA_128 = absCriterion:forward(rec[2], input_img_wrong_128) * opt.lambda1
  local df_do_cycle_128 = absCriterion:backward(rec[2], input_img_wrong_128):mul(opt.lambda1)
  local df_do_cycle_128_clone = df_do_cycle_128:clone()

  local df_do_rec = netG:updateGradInput({fake[2], input_txt_wrong}, {df_do_cycle_64_clone, df_do_cycle_128_clone})
  netG:backward({input_img_wrong_128, input_txt_real}, {df_d_GAN_64_clone + df_do_cycle_64_clone, df_d_GAN_128_clone + df_do_cycle_128_clone})

  errRecA = errRecA_64 + errRecA_128

  -- backward cycle loss
  local fake2 = netG:forward({input_img_real_128, input_txt_wrong})
  local rec2 = netG:forward({fake2[2], input_txt_real})
  errAdaptA_64 = absCriterion:forward(rec2[1], input_img_real) * opt.lambda2
  local df_do_coadapt_64 = absCriterion:backward(rec2[1], input_img_real):mul(opt.lambda2)
  local df_do_coadapt_64_clone = df_do_coadapt_64:clone()
  errAdaptA_128 = absCriterion:forward(rec2[2], input_img_real_128) * opt.lambda2
  local df_do_coadapt_128 = absCriterion:backward(rec2[2], input_img_real_128):mul(opt.lambda2)
  local df_do_coadapt_128_clone = df_do_coadapt_128:clone()
  netG:backward({fake2[2], input_txt_real}, {df_do_coadapt_64_clone, df_do_coadapt_128_clone})

  errAdaptA = errAdaptA_64 + errAdaptA_128

  return errGA, gradParametersG
end


local fDxB_64 = function(x)
  netD_64:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersD_64:zero()

  -- train with real
  label:fill(real_label)
  local output = netD_64:forward{input_img_wrong, input_txt_wrong}
  local errD_real_uncond = criterion:forward(output[1], label)
  local df_do_uncond = criterion:backward(output[1], label)
  local df_do_uncond_clone = df_do_uncond:clone()
  local errD_real = criterion:forward(output[2], label)
  local df_do = criterion:backward(output[2], label)
  local df_do_clone = df_do:clone()
  netD_64:backward({input_img_wrong, input_txt_wrong}, {df_do_uncond_clone, df_do_clone})

  -- train with wrong
  local output = netD_64:forward({input_img_real, input_txt_wrong})
  label:fill(real_label)
  local errD_wrong_uncond = opt.cls_weight * criterion:forward(output[1], label)
  local df_do_uncond = criterion:backward(output[1], label):mul(opt.cls_weight)
  local df_do_uncond_clone = df_do_uncond:clone()
  label:fill(fake_label)
  local errD_wrong = opt.cls_weight * criterion:forward(output[2], label)
  local df_do = criterion:backward(output[2], label):mul(opt.cls_weight)
  local df_do_clone = df_do:clone()
  netD_64:backward({input_img_real, input_txt_wrong}, {df_do_uncond_clone, df_do_clone})

  -- train with fake
  local fake_weight = 1 - opt.cls_weight
  local fake = netG:forward({input_img_real_128, input_txt_wrong})
  local output = netD_64:forward{fake[1], input_txt_wrong}
  label:fill(fake_label)
  local errD_fake_uncond = fake_weight * criterion:forward(output[1], label)
  local df_do_uncond = criterion:backward(output[1], label):mul(fake_weight)
  local df_do_uncond_clone = df_do_uncond:clone()
  local errD_fake = fake_weight * criterion:forward(output[2], label)
  local df_do = criterion:backward(output[2], label):mul(fake_weight)
  local df_do_clone = df_do:clone()
  netD_64:backward({fake, input_txt_wrong}, {df_do_uncond_clone, df_do_clone})

  errRB_64 = errD_real + errD_real_uncond
  errWB_64 = errD_wrong + errD_wrong_uncond
  errFB_64 = errD_fake + errD_fake_uncond
  errDB_64 = errRB_64 + errWB_64 + errFB_64

  return errDB_64, gradParametersD_64
end


local fDxB_128 = function(x)

  netD_128:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersD_128:zero()

  -- train with real
  label:fill(real_label)
  local output = netD_128:forward{input_img_wrong_128, input_txt_wrong}
  local errD_real_uncond = criterion:forward(output[1], label)
  local df_do_uncond = criterion:backward(output[1], label)
  local df_do_uncond_clone = df_do_uncond:clone()
  local errD_real = criterion:forward(output[2], label)
  local df_do = criterion:backward(output[2], label)
  local df_do_clone = df_do:clone()
  netD_128:backward({input_img_wrong_128, input_txt_wrong}, {df_do_uncond_clone, df_do_clone})

  -- train with wrong
  local output = netD_128:forward({input_img_real_128, input_txt_wrong})
  label:fill(real_label)
  local errD_wrong_uncond = opt.cls_weight * criterion:forward(output[1], label)
  local df_do_uncond = criterion:backward(output[1], label):mul(opt.cls_weight)
  local df_do_uncond_clone = df_do_uncond:clone()
  label:fill(fake_label)
  local errD_wrong = opt.cls_weight * criterion:forward(output[2], label)
  local df_do = criterion:backward(output[2], label):mul(opt.cls_weight)
  local df_do_clone = df_do:clone()
  netD_128:backward({input_img_real_128, input_txt_wrong}, {df_do_uncond_clone, df_do_clone})

  -- train with fake
  local fake_weight = 1 - opt.cls_weight
  local fake = netG:forward({input_img_real_128, input_txt_wrong})
  local output = netD_128:forward{fake[2], input_txt_wrong}
  label:fill(fake_label)
  local errD_fake_uncond = fake_weight * criterion:forward(output[1], label)
  local df_do_uncond = criterion:backward(output[1], label):mul(fake_weight)
  local df_do_uncond_clone = df_do_uncond:clone()
  local errD_fake = fake_weight * criterion:forward(output[2], label)
  local df_do = criterion:backward(output[2], label):mul(fake_weight)
  local df_do_clone = df_do:clone()
  netD_128:backward({fake, input_txt_wrong}, {df_do_uncond_clone, df_do_clone})

  errRB_128 = errD_real + errD_real_uncond
  errWB_128 = errD_wrong + errD_wrong_uncond
  errFB_128 = errD_fake + errD_fake_uncond
  errDB_128 = errRB_128 + errWB_128 + errFB_128

  return errDB_128, gradParametersD_128
end


-- create closure to evaluate f(X) and df/dX of generator
local fGxB = function(x)
  netD_64:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netD_128:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersG:zero()

  -- G should be identity if correspnding image and text pair is fed
  local identity = nil
  if opt.lambda_identity > 0 then
    identity = netG:forward({input_img_wrong_128, input_txt_wrong})
    local errI_64 = absCriterion:forward(identity[1], input_img_wrong) * lambda2 * opt.lambda_identity
    local didentity_loss_do_64 = absCriterion:backward(identity[1], input_img_wrong):mul(lambda2):mul(opt.lambda_identity)
    local didentity_loss_do_64_clone = didentity_loss_do_64:clone()
    local errI_128 = absCriterion:forward(identity[2], input_img_wrong_128) * lambda2 * opt.lambda_identity
    local didentity_loss_do_128 = absCriterion:backward(identity[2], input_img_wrong_128):mul(lambda2):mul(opt.lambda_identity)
    local didentity_loss_do_128_clone = didentity_loss_do_128:clone()
    netG:backward({input_img_wrong_128, input_txt_wrong}, {didentity_loss_do_64_clone, didentity_loss_do_128_clone})

    errIB = errI_64 + errI_128
  end

  -- GAN loss
  label:fill(real_label)
  local fake = netG:forward({input_img_real_128, input_txt_wrong})
  local output_64 = netD_64:forward({fake[1], input_txt_wrong})
  local errGB_64_uncond = criterion:forward(output_64[1], label)
  local df_do_64_uncond = criterion:backward(output_64[1], label)
  local df_do_64_uncond_clone = df_do_64_uncond:clone()
  local errGB_64 = criterion:forward(output_64[2], label)
  local df_do_64 = criterion:backward(output_64[2], label)
  local df_do_64_clone = df_do_64:clone()

  local output_128 = netD_128:forward({fake[2], input_txt_wrong})
  local errGB_128_uncond = criterion:forward(output_128[1], label)
  local df_do_128_uncond = criterion:backward(output_128[1], label)
  local df_do_128_uncond_clone = df_do_128_uncond:clone()
  local errGB_128 = criterion:forward(output_128[2], label)
  local df_do_128 = criterion:backward(output_128[2], label)
  local df_do_128_clone = df_do_128:clone()

  local df_d_GAN_64 = netD_64:updateGradInput({fake[1], input_txt_wrong}, {df_do_64_uncond_clone, df_do_64_clone})
  local df_d_GAN_64_clone = df_d_GAN_64[1]:clone()
  local df_d_GAN_128 = netD_128:updateGradInput({fake[2], input_txt_wrong}, {df_do_128_uncond_clone, df_do_128_clone})
  local df_d_GAN_128_clone = df_d_GAN_128[1]:clone()

  errGB = errGB_64_uncond + errGB_64 + errGB_128_uncond + errGB_128

  errRecB = -1
  errAdaptB = -1

  if errGB > opt.cycle_limit then
    netG:backward({input_img_wrong_128, input_txt_real}, {df_d_GAN_64_clone, df_d_GAN_128_clone})
    print("no cycle loss for GB")
    return errGB, gradParametersG
  end

  -- forward cycle loss
  local rec = netG:forward({fake[2], input_txt_real})
  local errRecB_64 = absCriterion:forward(rec[1], input_img_real) * opt.lambda1
  local df_do_cycle_64 = absCriterion:backward(rec[1], input_img_real):mul(opt.lambda1)
  local df_do_cycle_64_clone = df_do_cycle_64:clone()
  local errRecB_128 = absCriterion:forward(rec[2], input_img_real_128) * opt.lambda1
  local df_do_cycle_128 = absCriterion:backward(rec[2], input_img_real_128):mul(opt.lambda1)
  local df_do_cycle_128_clone = df_do_cycle_128:clone()

  local df_do_rec = netG:updateGradInput({fake[2], input_txt_real}, {df_do_cycle_64_clone, df_do_cycle_128_clone})
  netG:backward({input_img_real_128, input_txt_wrong}, {df_d_GAN_64_clone + df_do_cycle_64_clone, df_d_GAN_128_clone + df_do_cycle_128_clone})

  errRecB = errRecB_64 + errRecB_128

  -- backward cycle loss
  local fake2 = netG:forward({input_img_wrong_128, input_txt_real})
  local rec2 = netG:forward({fake2[2], input_txt_wrong})
  errAdaptB_64 = absCriterion:forward(rec2[1], input_img_wrong) * opt.lambda2
  local df_do_coadapt_64 = absCriterion:backward(rec2[1], input_img_wrong):mul(opt.lambda2)
  local df_do_coadapt_64_clone = df_do_coadapt_64:clone()
  errAdaptB_128 = absCriterion:forward(rec2[2], input_img_wrong_128) * opt.lambda2
  local df_do_coadapt_128 = absCriterion:backward(rec2[2], input_img_wrong_128):mul(opt.lambda2)
  local df_do_coadapt_128_clone = df_do_coadapt_128:clone()
  netG:backward({fake2[2], input_txt_wrong}, {df_do_coadapt_64_clone, df_do_coadapt_128_clone})

  return errGB, gradParametersG
end


-- train
for epoch = opt.epoch_begin, opt.niter do
  epoch_tm:reset()
  if epoch == 1 then
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. 0 .. '_net_G.t7', netG)
  end
  if epoch >= 1000 and epoch % 20 == 0 then
    optimStateG.learningRate = optimStateG.learningRate * opt.lr_decay
    optimStateD_64.learningRate = optimStateD_64.learningRate * opt.lr_decay
    optimStateD_128.learningRate = optimStateD_128.learningRate * opt.lr_decay
  end

  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
    tm:reset()
    data_tm:reset(); data_tm:resume()
    real_img, real_txt, wrong_img, wrong_txt, real_img_128, wrong_img_128 = data:getBatch()
    data_tm:stop()

    input_img_real:copy(real_img)
    input_img_wrong:copy(wrong_img)
    input_txt_real_raw:copy(real_txt)
    input_txt_wrong_raw:copy(wrong_txt)
    input_img_real_128:copy(real_img_128)
    input_img_wrong_128:copy(wrong_img_128)

    -- average adjacent text features in batch dimension.
    emb_txt_real = net_txt:forward(input_txt_real_raw)
    input_txt_real:copy(emb_txt_real)
    emb_txt_wrong = net_txt:forward(input_txt_wrong_raw)
    input_txt_wrong:copy(emb_txt_wrong)

    optim.adam(fDxA_64, parametersD_64, optimStateD_64)
    optim.adam(fDxA_128, parametersD_128, optimStateD_128)
    optim.adam(fGxA, parametersG, optimStateG)
    optim.adam(fDxB_64, parametersD_64, optimStateD_64)
    optim.adam(fDxB_128, parametersD_128, optimStateD_128)
    optim.adam(fGxB, parametersG, optimStateG)

    -- logging
    if ((i-1) / opt.batchSize) % opt.print_every == 0 then
    print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                .. '  Err_G: %.4f Err_D_64: %.4f Err_D_128: %.4f Err_Rec: %.4f Err_Adapt: %.4f Err_R_64: %.4f Err_W_64: %.4f Err_F_64: %.4f Err_R_128: %.4f Err_W_128: %.4f Err_F_128: %.4f Err_I: %.4f'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateG.learningRate,
              errGA and errGA or -1,
              errDA_64 and errDA_64 or -1,
              errDA_128 and errDA_128 or -1,
              errRecA and errRecA or -1,
              errAdaptA and errAdaptA or -1,
              errRA_64 and errRA_64 or -1,
              errWA_64 and errWA_64 or -1,
              errFA_64 and errFA_64 or -1,
              errRA_128 and errRA_128 or -1,
              errWA_128 and errWA_128 or -1,
              errFA_128 and errFA_128 or -1,
              errIA and errIA or -1))

    print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                .. '  Err_G: %.4f Err_D_64: %.4f Err_D_128: %.4f Err_Rec: %.4f Err_Adapt: %.4f Err_R_64: %.4f Err_W_64: %.4f Err_F_64: %.4f Err_R_128: %.4f Err_W_128: %.4f Err_F_128: %.4f Err_I: %.4f'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateG.learningRate,
              errGB and errGB or -1,
              errDB_64 and errDB_64 or -1,
              errDB_128 and errDB_128 or -1,
              errRecB and errRecB or -1,
              errAdaptB and errAdaptB or -1,
              errRB_64 and errRB_64 or -1,
              errWB_64 and errWB_64 or -1,
              errFB_64 and errFB_64 or -1,
              errRB_128 and errRB_128 or -1,
              errWB_128 and errWB_128 or -1,
              errFB_128 and errFB_128 or -1,
              errIB and errIB or -1))

      local fake = netG.output
      disp.image(fake[2]:narrow(1,1,opt.batchSize), {win=opt.display_id, title=opt.name})
      disp.image(real_img, {win=opt.display_id * 3, title=opt.name})
    end
  end

  -- save checkpoints
  if epoch % opt.save_every == 0 then
    paths.mkdir(opt.checkpoint_dir)
    if opt.stage1 == 1 then
      torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG)
      torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_D_64.t7', netD_64)
      torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_D_128.t7', netD_128)
    end
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_opt.t7', opt)
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
           epoch, opt.niter, epoch_tm:time().real))
  end
end
