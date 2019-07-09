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
   init_g_2 = '',
   init_d = '/home/xhy/code/textEditImage/checkpoints_cub2/netD.t7',
   init_d_2 = '',
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
netD = torch.load(opt.init_d)
if opt.stage2 == 1 then
  netG_stage2 = torch.load(opt.init_g_2)
  netD_stage2 = torch.load(opt.init_d_2)
end
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
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateG_stage2 = {
   learningRate = opt.lr_stage2,
   beta1 = opt.beta1,
}
optimStateD_stage2 = {
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
    netD:cuda()
    netG:cuda()
    if opt.stage2 == 1 then
        netD_stage2:cuda()
        netG_stage2:cuda()
    end
    net_txt:cuda()
    criterion:cuda()
    absCriterion:cuda()
    mseCriterion:cuda()
end

if opt.use_cudnn == 1 then
  cudnn = require('cudnn')
  netD = cudnn.convert(netD, cudnn)
  netG = cudnn.convert(netG, cudnn)
  net_txt = cudnn.convert(net_txt, cudnn)
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.stage2 == 1 then
  parametersD_stage2, gradParametersD_stage2 = netD_stage2:getParameters()
  parametersG_stage2, gradParametersG_stage2 = netG_stage2:getParameters()
end

if opt.display then disp = require 'display' end

local fDxA = function(x)
  netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersD:zero()

  -- train with real
  label:fill(real_label)
  local output = netD:forward{input_img_real, input_txt_real}
  errD_real = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netD:backward({input_img_real, input_txt_real}, df_do)

  -- train with wrong
  errD_wrong = 0
  label:fill(fake_label)
  local output = netD:forward({input_img_wrong, input_txt_real})
  errD_wrong = opt.cls_weight * criterion:forward(output, label)
  local df_do = criterion:backward(output, label):mul(opt.cls_weight)
  netD:backward({input_img_wrong, input_txt_real}, df_do)

  -- train with fake
  local fake_weight = 1 - opt.cls_weight

  --[[
  local fake = netG:forward({input_img_real, input_txt_real})
  label:fill(fake_label)

  local output = netD:forward{fake, input_txt_real}
  local errD_fake1 = fake_weight * criterion:forward(output, label)
  local df_do = criterion:backward(output, label):mul(fake_weight)
  netD:backward({fake, input_txt_real}, df_do)
  --]]

  local fake = netG:forward({input_img_wrong, input_txt_real})
  label:fill(fake_label)

  local output = netD:forward{fake, input_txt_real}
  local errD_fake2 = fake_weight * criterion:forward(output, label)
  local df_do = criterion:backward(output, label):mul(fake_weight)
  netD:backward({fake, input_txt_real}, df_do)

  errRA = errD_real
  errWA = errD_wrong
  errFA = -1
  errFA = errD_fake2
  errDA = errRA + errWA + errFA

  return errDA, gradParametersD
end


-- create closure to evaluate f(X) and df/dX of generator
local fGxA = function(x)
  netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersG:zero()

  errIA = -1
  errGA = -1
  errAdaptA = -1
  errRecA = -1

  -- G should be identity if corresponding image and text pair is fed
  local identity = nil
  if opt.lambda_identity > 0 then
    identity = netG:forward({input_img_real, input_txt_real}):clone()
    errIx = absCriterion:forward(identity, input_img_real) * lambda2 * opt.lambda_identity
    local didentity_loss_do = absCriterion:backward(identity, input_img_real):mul(lambda2):mul(opt.lambda_identity)
    netG:backward({input_img_real, input_txt_real}, didentity_loss_do)

--[[
    label:fill(real_label)
    output = netD:forward({identity, input_txt_real})
    errT1= criterion:forward(output, label)
    df_do = criterion:backward(output, label)
    df_dg = netD:updateGradInput({identity, input_txt_real}, df_do)
    netG:backward({input_img_real, input_txt_real}, df_dg[1])

    local identity = netG:forward({input_img_real, input_txt_wrong})
    errIy = absCriterion:forward(identity, input_img_real)
    local didentity_loss_do = absCriterion:backward(identity, input_img_real)
    netG:backward({input_img_real, input_txt_wrong}, didentity_loss_do)
--]]

    errIA = errIx
  end

  -- GAN loss
  label:fill(real_label)
  local fake = netG:forward({input_img_wrong, input_txt_real})
  local output = netD:forward({fake, input_txt_real})
  errGA = criterion:forward(output, label)
  local df_do1 = criterion:backward(output, label)
  local df_d_GAN = netD:updateGradInput({fake, input_txt_real}, df_do1)

  if errGA > opt.cycle_limit then
    netG:backward({input_img_wrong, input_txt_real}, df_d_GAN[1])
    print("no cycle loss for GA")
    return errGA, gradParametersG
  end

  -- forward cycle loss
  local rec = netG:forward({fake, input_txt_wrong})
  errRecA = absCriterion:forward(rec, input_img_wrong) * opt.lambda1
  local df_do2 = absCriterion:backward(rec, input_img_wrong):mul(opt.lambda1)
  local df_do_rec = netG:updateGradInput({fake, input_txt_wrong}, df_do2)

  netG:backward({input_img_wrong, input_txt_real}, df_d_GAN[1] + df_do_rec[1])

  -- backward cycle loss
  local fake2 = netG:forward({input_img_real, input_txt_wrong})
  local rec2 = netG:forward({fake2, input_txt_real})
  errAdaptA = absCriterion:forward(rec2, input_img_real) * opt.lambda2
  local df_do_coadapt = absCriterion:backward(rec2, input_img_real):mul(opt.lambda2)
  netG:backward({fake2, input_txt_real}, df_do_coadapt)

  return errGA, gradParametersG
end


local fDxB = function(x)
  netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersD:zero()

  -- train with real
  label:fill(real_label)
  local output = netD:forward{input_img_wrong, input_txt_wrong}
  errD_real = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netD:backward({input_img_wrong, input_txt_wrong}, df_do)

  -- train with wrong
  errD_wrong = 0
  label:fill(fake_label)
  local output = netD:forward({input_img_real, input_txt_wrong})
  errD_wrong = opt.cls_weight * criterion:forward(output, label)
  local df_do = criterion:backward(output, label):mul(opt.cls_weight)
  netD:backward({input_img_real, input_txt_wrong}, df_do)

  -- train with fake
  local fake_weight = 1 - opt.cls_weight

  --[[
  local fake = netG:forward({input_img_wrong, input_txt_wrong})
  label:fill(fake_label)

  local output = netD:forward{fake, input_txt_wrong}
  local errD_fake1 = fake_weight * criterion:forward(output, label)
  local df_do = criterion:backward(output, label):mul(fake_weight)
  netD:backward({fake, input_txt_wrong}, df_do)
  --]]

  local fake = netG:forward({input_img_real, input_txt_wrong})
  label:fill(fake_label)

  local output = netD:forward{fake, input_txt_wrong}
  local errD_fake2 = fake_weight * criterion:forward(output, label)
  local df_do = criterion:backward(output, label):mul(fake_weight)
  netD:backward({fake, input_txt_wrong}, df_do)

  errRB = errD_real
  errWB = errD_wrong
  errFB = -1
  errFB = errD_fake2
  errDB = errRB + errWB + errFB

  return errDB, gradParametersD
end


-- create closure to evaluate f(X) and df/dX of generator
local fGxB = function(x)
  netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersG:zero()

  errIB = -1
  errGB = -1
  errAdaptB = -1
  errRecB = -1

  -- G should be identity if correspnding image and text pair is fed
  local identity = nil
  if opt.lambda_identity > 0 then
    identity = netG:forward({input_img_wrong, input_txt_wrong}):clone()
    errIx = absCriterion:forward(identity, input_img_wrong) * lambda2 * opt.lambda_identity
    local didentity_loss_do = absCriterion:backward(identity, input_img_wrong):mul(lambda2):mul(opt.lambda_identity)
    netG:backward({input_img_wrong, input_txt_wrong}, didentity_loss_do)

--[[
    label:fill(real_label)
    output = netD:forward({identity, input_txt_real})
    errT2= criterion:forward(output, label)
    df_do = criterion:backward(output, label)
    df_dg = netD:updateGradInput({identity, input_txt_real}, df_do)
    netG:backward({input_img_real, input_txt_real}, df_dg[1])

    local identity = netG:forward({input_img_real, input_txt_wrong})
    errIy = absCriterion:forward(identity, input_img_real)
    local didentity_loss_do = absCriterion:backward(identity, input_img_real)
    netG:backward({input_img_real, input_txt_wrong}, didentity_loss_do)
--]]

    errIB = errIx

  end

  -- GAN loss
  label:fill(real_label)
  local fake = netG:forward({input_img_real, input_txt_wrong})
  local output = netD:forward({fake, input_txt_wrong})
  errGB = criterion:forward(output, label)
  local df_do1 = criterion:backward(output, label)
  local df_d_GAN = netD:updateGradInput({fake, input_txt_wrong}, df_do1)

  if errGB > opt.cycle_limit then
    netG:backward({input_img_real, input_txt_wrong}, df_d_GAN[1])
    print("no cycle loss for GB")
    return errGB, gradParametersG
  end

  -- forward cycle loss
  local rec = netG:forward({fake, input_txt_real})
  errRecB = absCriterion:forward(rec, input_img_real) * opt.lambda1
  local df_do2 = absCriterion:backward(rec, input_img_real):mul(opt.lambda1)
  local df_do_rec = netG:updateGradInput({fake, input_txt_real}, df_do2)

  netG:backward({input_img_real, input_txt_wrong}, df_d_GAN[1] + df_do_rec[1])

  -- backward cycle loss
  local fake2 = netG:forward({input_img_wrong, input_txt_real})
  local rec2 = netG:forward({fake2, input_txt_wrong})
  errAdaptB = absCriterion:forward(rec2, input_img_wrong) * opt.lambda2
  local df_do_coadapt = absCriterion:backward(rec2, input_img_wrong):mul(opt.lambda2)
  netG:backward({fake2, input_txt_wrong}, df_do_coadapt)

  return errGB, gradParametersG
end


local fDxA2 = function(x)
  netD_stage2:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG_stage2:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersD_stage2:zero()

  -- train with real
  label:fill(real_label)
  local output = netD_stage2:forward{input_img_real_128, input_txt_real}
  errD_real_stage2 = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netD_stage2:backward({input_img_real_128, input_txt_real}, df_do)

  -- train with wrong
  errD_wrong = 0
  label:fill(fake_label)
  local output = netD_stage2:forward({input_img_wrong_128, input_txt_real})
  errD_wrong_stage2 = opt.cls_weight * criterion:forward(output, label)
  local df_do = criterion:backward(output, label):mul(opt.cls_weight)
  netD_stage2:backward({input_img_wrong_128, input_txt_real}, df_do)

  -- train with fake
  local fake_weight = 1 - opt.cls_weight
  label:fill(fake_label)
  local fake64 = netG:forward({input_img_wrong, input_txt_real})
  local fake128 = netG_stage2:forward({{input_img_wrong_128, fake64}, input_txt_real})

  local output = netD_stage2:forward{fake128, input_txt_real}
  local errD_fake2_stage2 = fake_weight * criterion:forward(output, label)
  local df_do = criterion:backward(output, label):mul(fake_weight)
  netD_stage2:backward({fake128, input_txt_real}, df_do)

  errRA_stage2 = errD_real_stage2
  errWA_stage2 = errD_wrong_stage2
  errFA_stage2 = errD_fake2_stage2
  errDA_stage2 = errRA_stage2 + errWA_stage2 + errFA_stage2

  return errDA_stage2, gradParametersD_stage2
end

local fGxA2 = function(x)
  netD_stage2:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG_stage2:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersG_stage2:zero()

  errIA_stage2 = -1
  errGA_stage2 = -1
  errAdaptA_stage2 = -1
  errRecA_stage2 = -1

  -- G should be identity if corresponding image and text pair is fed
  local identity = nil
  if opt.lambda_identity > 0 then
    local identity64 = netG:forward({input_img_real, input_txt_real})
    local identity128 = netG_stage2:forward({{input_img_real_128, identity64}, input_txt_real})
    local errIx_stage2 = absCriterion:forward(identity128, input_img_real_128) * lambda2 * opt.lambda_identity
    local didentity_loss_do = absCriterion:backward(identity128, input_img_real_128):mul(lambda2):mul(opt.lambda_identity)
    netG_stage2:backward({{input_img_real_128, identity64}, input_txt_real}, didentity_loss_do)
    errIA_stage2 = errIx_stage2
  end

  -- GAN loss
  label:fill(real_label)
  local fake64 = netG:forward({input_img_wrong, input_txt_real})
  local fake128 = netG_stage2:forward({{input_img_wrong_128, fake64}, input_txt_real})
  local output = netD_stage2:forward({fake128, input_txt_real})
  errGA_stage2 = criterion:forward(output, label)
  local df_do1 = criterion:backward(output, label)
  local df_d_GAN = netD_stage2:updateGradInput({fake128, input_txt_real}, df_do1)


  if errGA_stage2 > opt.cycle_limit_stage2 then
    netG_stage2:backward({{input_img_wrong_128, fake64}, input_txt_real}, df_d_GAN[1])
    print("no cycle loss for GA stage2")
    return errGA_stage2, gradParametersG_stage2
  end


  -- forward cycle loss
  local fake128_64 = torch.Tensor(opt.batchSize, 3, 64, 64)
  for i=1, opt.batchSize do
    fake128_64[i] = image.scale(fake128:narrow(1, i, 1):view(3, 128, 128):float(), 64, 64)
  end
  fake128_64 = fake128_64:cuda()
  local rec64 = netG:forward({fake128_64, input_txt_wrong})
  local rec128 = netG_stage2:forward({{fake128, rec64}, input_txt_wrong})
  errRecA_stage2 = absCriterion:forward(rec128, input_img_wrong_128) * opt.lambda1
  local df_do2 = absCriterion:backward(rec128, input_img_wrong_128):mul(opt.lambda1)
  local df_do_rec1 = netG_stage2:updateGradInput({{fake128, rec64}, input_txt_wrong}, df_do2)
  local df_do_rec2 = netG:updateGradInput({fake128_64, input_txt_wrong}, df_do_rec1[1][2])
  local df_do_rec2_1 = torch.Tensor(opt.batchSize, 3, 128, 128)
  for i=1, opt.batchSize do
    df_do_rec2_1[i] = image.scale(df_do_rec2[1]:narrow(1, i, 1):view(3, 64, 64):float(), 128, 128)
  end
  df_do_rec2_1 = df_do_rec2_1:cuda()

  netG_stage2:backward({{input_img_wrong_128, fake64}, input_txt_real}, df_d_GAN[1] + df_do_rec2_1)

  -- backward cycle loss
  local fake64_2 = netG:forward({input_img_real, input_txt_wrong})
  local fake128_2 = netG_stage2:forward({{input_img_real_128, fake64_2}, input_txt_wrong})
  local fake128_2_64 = torch.Tensor(opt.batchSize, 3, 64, 64)
  for i=1, opt.batchSize do
    fake128_2_64[i] = image.scale(fake128_2:narrow(1, i, 1):view(3, 128, 128):float(), 64, 64)
  end
  fake128_2_64 = fake128_2_64:cuda()
  local rec64_2 = netG:forward({fake128_2_64, input_txt_real})
  local rec128_2 = netG_stage2:forward({{fake128_2, rec64_2}, input_txt_real})
  errAdaptA_stage2 = absCriterion:forward(rec128_2, input_img_real_128) * opt.lambda2
  local df_do_coadapt = absCriterion:backward(rec128_2, input_img_real_128):mul(opt.lambda2)
  netG_stage2:backward({{fake128_2, rec64_2}, input_txt_real}, df_do_coadapt)

  return errGA_stage2, gradParametersG_stage2
end


local fDxB2 = function(x)
  netD_stage2:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG_stage2:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersD_stage2:zero()

  -- train with real
  label:fill(real_label)
  local output = netD_stage2:forward({input_img_wrong_128, input_txt_wrong})
  errD_real_stage2 = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netD_stage2:backward({input_img_wrong_128, input_txt_wrong}, df_do)

  -- train with wrong
  errD_wrong = 0
  label:fill(fake_label)
  local output = netD_stage2:forward({input_img_real_128, input_txt_wrong})
  errD_wrong_stage2 = opt.cls_weight * criterion:forward(output, label)
  local df_do = criterion:backward(output, label):mul(opt.cls_weight)
  netD_stage2:backward({input_img_real_128, input_txt_wrong}, df_do)

  -- train with fake
  local fake_weight = 1 - opt.cls_weight
  label:fill(fake_label)
  local fake64 = netG:forward({input_img_real, input_txt_wrong})
  local fake128 = netG_stage2:forward({{input_img_real_128, fake64}, input_txt_wrong})

  local output = netD_stage2:forward{fake128, input_txt_wrong}
  local errD_fake2_stage2 = fake_weight * criterion:forward(output, label)
  local df_do = criterion:backward(output, label):mul(fake_weight)
  netD_stage2:backward({fake128, input_txt_wrong}, df_do)

  errRB_stage2 = errD_real_stage2
  errWB_stage2 = errD_wrong_stage2
  errFB_stage2 = errD_fake2_stage2
  errDB_stage2 = errRB_stage2 + errWB_stage2 + errFB_stage2

  return errDB_stage2, gradParametersD_stage2
end

local fGxB2 = function(x)
  netD_stage2:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG_stage2:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersG_stage2:zero()

  errIB_stage2 = -1
  errGB_stage2 = -1
  errAdaptB_stage2 = -1
  errRecB_stage2 = -1

  -- G should be identity if corresponding image and text pair is fed
  local identity = nil
  if opt.lambda_identity > 0 then
    local identity64 = netG:forward({input_img_wrong, input_txt_wrong})
    local identity128 = netG_stage2:forward({{input_img_wrong_128, identity64}, input_txt_wrong})
    local errIx_stage2 = absCriterion:forward(identity128, input_img_wrong_128) * lambda2 * opt.lambda_identity
    local didentity_loss_do = absCriterion:backward(identity128, input_img_wrong_128):mul(lambda2):mul(opt.lambda_identity)
    netG_stage2:backward({{input_img_wrong_128, identity64}, input_txt_wrong}, didentity_loss_do)
    errIB_stage2 = errIx_stage2
  end

  -- GAN loss
  label:fill(real_label)
  local fake64 = netG:forward({input_img_real, input_txt_wrong})
  local fake128 = netG_stage2:forward({{input_img_real_128, fake64}, input_txt_wrong})
  local output = netD_stage2:forward({fake128, input_txt_wrong})
  errGB_stage2 = criterion:forward(output, label)
  local df_do1 = criterion:backward(output, label)
  local df_d_GAN = netD_stage2:updateGradInput({fake128, input_txt_wrong}, df_do1)

  if errGB_stage2 > opt.cycle_limit_stage2 then
    netG_stage2:backward({{input_img_real_128, fake64}, input_txt_wrong}, df_d_GAN[1])
    print("no cycle loss for GB stage2")
    return errGB_stage2, gradParametersG_stage2
  end

  -- forward cycle loss
  local fake128_64 = torch.Tensor(opt.batchSize, 3, 64, 64)
  for i=1, opt.batchSize do
    fake128_64[i] = image.scale(fake128:narrow(1, i, 1):view(3, 128, 128):float(), 64, 64)
  end
  fake128_64 = fake128_64:cuda()
  local rec64 = netG:forward({fake128_64, input_txt_real})
  local rec128 = netG_stage2:forward({{fake128, rec64}, input_txt_real})
  errRecB_stage2 = absCriterion:forward(rec128, input_img_real_128) * opt.lambda1
  local df_do2 = absCriterion:backward(rec128, input_img_real_128):mul(opt.lambda1)
  local df_do_rec1 = netG_stage2:updateGradInput({{fake128, rec64}, input_txt_real}, df_do2)
  local df_do_rec2 = netG:updateGradInput({fake128_64, input_txt_real}, df_do_rec1[1][2])
  local df_do_rec2_1 = torch.Tensor(opt.batchSize, 3, 128, 128)
  for i=1, opt.batchSize do
    df_do_rec2_1[i] = image.scale(df_do_rec2[1]:narrow(1, i, 1):view(3, 64, 64):float(), 128, 128)
  end
  df_do_rec2_1 = df_do_rec2_1:cuda()

  netG_stage2:backward({{input_img_real_128, fake64}, input_txt_wrong}, df_d_GAN[1] + df_do_rec2_1)

  -- backward cycle loss
  local fake64_2 = netG:forward({input_img_wrong, input_txt_real})
  local fake128_2 = netG_stage2:forward({{input_img_wrong_128, fake64_2}, input_txt_real})
  local fake128_2_64 = torch.Tensor(opt.batchSize, 3, 64, 64)
  for i=1, opt.batchSize do
    fake128_2_64[i] = image.scale(fake128_2:narrow(1, i, 1):view(3, 128, 128):float(), 64, 64)
  end
  fake128_2_64 = fake128_2_64:cuda()
  local rec64_2 = netG:forward({fake128_2_64, input_txt_wrong})
  local rec128_2 = netG_stage2:forward({{fake128_2, rec64_2}, input_txt_wrong})
  errAdaptB_stage2 = absCriterion:forward(rec128_2, input_img_wrong_128) * opt.lambda2
  local df_do_coadapt = absCriterion:backward(rec128_2, input_img_wrong_128):mul(opt.lambda2)
  netG_stage2:backward({{fake128_2, rec64_2}, input_txt_wrong}, df_do_coadapt)

  return errGB_stage2, gradParametersG_stage2
end


-- train
for epoch = opt.epoch_begin, opt.niter do
  epoch_tm:reset()
  if epoch == 1 then
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. 0 .. '_net_G.t7', netG)
  end
  if epoch >= 1000 and epoch % 20 == 0 then
    optimStateG.learningRate = optimStateG.learningRate * opt.lr_decay
    optimStateD.learningRate = optimStateD.learningRate * opt.lr_decay
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

    if opt.stage1 == 1 then
      optim.adam(fDxA, parametersD, optimStateD)
      optim.adam(fGxA, parametersG, optimStateG)
    end
    if opt.stage2 == 1 then
      optim.adam(fDxA2, parametersD_stage2, optimStateD_stage2)
      optim.adam(fGxA2, parametersG_stage2, optimStateG_stage2)
    end
    if opt.stage1 == 1 then
      optim.adam(fDxB, parametersD, optimStateD)
      optim.adam(fGxB, parametersG, optimStateG)
    end
    if opt.stage2 == 1 then
      optim.adam(fDxB2, parametersD_stage2, optimStateD_stage2)
      optim.adam(fGxB2, parametersG_stage2, optimStateG_stage2)
    end


    -- logging
    if ((i-1) / opt.batchSize) % opt.print_every == 0 then
      if opt.stage1 == 1 then
        print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                .. '  Err_G: %.4f Err_D: %.4f Err_Rec: %.4f Err_Adapt: %.4f Err_R: %.4f Err_W: %.4f Err_F: %.4f Err_I: %.4f'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateG.learningRate,
              errGA and errGA or -1,
              errDA and errDA or -1,
              errRecA and errRecA or -1,
              errAdaptA and errAdaptA or -1,
              errRA and errRA or -1,
              errWA and errWA or -1,
              errFA and errFA or -1,
              errIA and errIA or -1))
      end
      if opt.stage2 == 1 then
        print(('[%d][%d/%d] T:%.3f  DT:%.3f lr_stage2: %.4g '
                .. '  Err_G2: %.4f Err_D2: %.4f Err_Rec2: %.4f Err_Adapt2: %.4f Err_R2: %.4f Err_W2: %.4f Err_F2: %.4f Err_I2: %.4f'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateG_stage2.learningRate,
              errGA_stage2 and errGA_stage2 or -1,
              errDA_stage2 and errDA_stage2 or -1,
              errRecA_stage2 and errRecA_stage2 or -1,
              errAdaptA_stage2 and errAdaptA_stage2 or -1,
              errRA_stage2 and errRA_stage2 or -1,
              errWA_stage2 and errWA_stage2 or -1,
              errFA_stage2 and errFA_stage2 or -1,
              errIA_stage2 and errIA_stage2 or -1))
      end
      if opt.stage1 == 1 then
        print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                .. '  Err_G: %.4f Err_D: %.4f Err_Rec: %.4f Err_Adapt: %.4f Err_R: %.4f Err_W: %.4f Err_F: %.4f Err_I: %.4f'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateG.learningRate,
              errGB and errGB or -1,
              errDB and errDB or -1,
              errRecB and errRecB or -1,
              errAdaptB and errAdaptB or -1,
              errRB and errRB or -1,
              errWB and errWB or -1,
              errFB and errFB or -1,
              errIB and errIB or -1))
      end
      if opt.stage2 == 1 then
        print(('[%d][%d/%d] T:%.3f  DT:%.3f lr_stage2: %.4g '
                .. '  Err_G2: %.4f Err_D2: %.4f Err_Rec2: %.4f Err_Adapt2: %.4f Err_R2: %.4f Err_W2: %.4f Err_F2: %.4f Err_I2: %.4f'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateG_stage2.learningRate,
              errGB_stage2 and errGB_stage2 or -1,
              errDB_stage2 and errDB_stage2 or -1,
              errRecB_stage2 and errRecB_stage2 or -1,
              errAdaptB_stage2 and errAdaptB_stage2 or -1,
              errRB_stage2 and errRB_stage2 or -1,
              errWB_stage2 and errWB_stage2 or -1,
              errFB_stage2 and errFB_stage2 or -1,
              errIB_stage2 and errIB_stage2 or -1))
      end
      local fake = netG.output
      disp.image(fake:narrow(1,1,opt.batchSize), {win=opt.display_id, title=opt.name})
      disp.image(real_img, {win=opt.display_id * 3, title=opt.name})
    end
  end

  -- save checkpoints
  if epoch % opt.save_every == 0 then
    paths.mkdir(opt.checkpoint_dir)
    if opt.stage1 == 1 then
      torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG)
      torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD)
    end
    if opt.stage2 == 1 then
      torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_G_stage2.t7', netG_stage2)
      torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_D_stage2.t7', netD_stage2)
    end
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_opt.t7', opt)
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
           epoch, opt.niter, epoch_tm:time().real))
  end
end
