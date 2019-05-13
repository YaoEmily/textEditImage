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
   checkpoint_dir = '/home/xhy/code/textEditImage/checkpoints',
   numshot = 0,
   batchSize = 64,
   doc_length = 201,
   loadSize = 76,
   txtSize = 1024,         -- #  of dim for raw text.
   fineSize = 64,
   nt = 128,               -- #  of dim for text features.
   nz = 10,               -- #  of dim for Z
   ngf = 128,              -- #  of gen filters in first conv layer
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

   init_g = '',
   init_d = '',
   net_txt = '/home/xhy/code/textEditImage/dataset_cub/lm_sje_nc4_cub_hybrid_gru18_a1_c512_0.00070_1_10_trainvalids.txt_iter30000.t7',
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

local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

netG = torch.load(opt.init_g)
netD = torch.load(opt.init_d)
--netG:apply(weights_init)
--print(netG)
--print(netD)

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

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
alphabet_size = #alphabet
local input_img = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input_img_real = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input_img_wrong = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
if opt.replicate == 1 then
  input_txt_real_raw = torch.Tensor(opt.batchSize, opt.doc_length, #alphabet)
  input_txt_wrong_raw = torch.Tensor(opt.batchSize, opt.doc_length, #alphabet)
else
  input_txt_real_raw = torch.Tensor(opt.batchSize * opt.numCaption, opt.txtSize)
  input_txt_wrong_raw = torch.Tensor(opt.batchSize * opt.numCaption, opt.txtSize)
end
local input_txt_real = torch.Tensor(opt.batchSize, opt.txtSize)
local input_txt_wrong = torch.Tensor(opt.batchSize, opt.txtSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG, errR, errW, errF, errA, errRec, errAdapt
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

if opt.gpu > 0 then
   input_img = input_img:cuda()
   input_img_real = input_img_real:cuda()
   input_img_wrong = input_img_wrong:cuda()
   input_txt_real = input_txt_real:cuda()
   input_txt_wrong = input_txt_wrong:cuda()
   input_txt_real_raw = input_txt_real_raw:cuda()
   input_txt_wrong_raw = input_txt_wrong_raw:cuda()
   noise = noise:cuda()
   label = label:cuda()
   netD:cuda()
   netG:cuda()
   --netR:cuda()
   net_txt:cuda()
   criterion:cuda()
   absCriterion:cuda()
   mseCriterion:cuda()
end

if opt.use_cudnn == 1 then
  cudnn = require('cudnn')
  netD = cudnn.convert(netD, cudnn)
  netG = cudnn.convert(netG, cudnn)
  --netR = cudnn.convert(netR, cudnn)
  net_txt = cudnn.convert(net_txt, cudnn)
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end



local fDx = function(x)
  netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') and m.bias~=nil then m.bias:zero() end end)

  gradParametersD:zero()

  -- train with real
  data_tm:reset(); data_tm:resume()
  real_img, real_txt, wrong_img, wrong_txt = data:getBatch()
  data_tm:stop()

  input_img:copy(real_img)
  input_img_real:copy(real_img)
  input_img_wrong:copy(wrong_img)
  input_txt_real_raw:copy(real_txt)
  input_txt_wrong_raw:copy(wrong_txt)

  -- average adjacent text features in batch dimension.
  emb_txt_real = net_txt:forward(input_txt_real_raw)
  input_txt_real:copy(emb_txt_real)
  emb_txt_wrong = net_txt:forward(input_txt_wrong_raw)
  input_txt_wrong:copy(emb_txt_wrong)

  label:fill(real_label)

  local output = netD:forward{input_img, input_txt_real}
  local errD_real = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netD:backward({input_img, input_txt_real}, df_do)

  -- train with wrong
  errD_wrong = 0
  if opt.cls_weight > 0 then
    -- train with wrong
    label:fill(fake_label)

    local output = netD:forward({input_img_wrong, input_txt_real})
    errD_wrong = opt.cls_weight*criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    df_do:mul(opt.cls_weight)
    netD:backward({input_img_wrong, input_txt_real}, df_do)
  end

  -- train with fake
  local fake = netG:forward{input_img_wrong, input_txt_real} -- wrong image + real text
  input_img:copy(fake)
  label:fill(fake_label)

  local output = netD:forward({input_img, input_txt_real})
  local errD_fake = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local fake_weight = 1 - opt.cls_weight
  errD_fake = errD_fake*fake_weight
  df_do:mul(fake_weight)
  netD:backward(({input_img, input_txt_real}), df_do)

  errD = errD_real + errD_fake + errD_wrong
  errW = errD_wrong
  errR = errD_real
  errF = errD_fake

  return errD, gradParametersD
end



-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
  netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') and m.bias~=nil then m.bias:zero() end end)

  gradParametersG:zero()

  -- GAN loss
  label:fill(real_label)
  local fake = netG:forward({input_img_wrong, input_txt_real})
  local output = netD:forward({fake, input_txt_real})
  errG = criterion:forward(output, label)
  local df_do1 = criterion:backward(output, label)
  local df_d_GAN = netD:updateGradInput({fake, input_txt_real}, df_do1)

  -- forward cycle loss
  local rec = netG:forward({fake, input_txt_wrong})
  errRec = mseCriterion:forward(rec, input_img_wrong)
  local df_do2 = mseCriterion:backward(rec, input_img_wrong)
  local df_do_rec = netG:updateGradInput({fake, input_txt_wrong}, df_do2)
  netG:backward({input_img_wrong, input_txt_real}, df_d_GAN[1] + df_do_rec[1])

  -- backward cycle loss
  local fake2 = netG:forward({input_img_real, input_txt_wrong})
  local rec2 = netG:forward({fake2, input_txt_real})
  errAdapt = mseCriterion:forward(rec2, input_img_real)
  local df_do_coadapt = mseCriterion:backward(rec2, input_img_real)
  netG:backward({fake2, input_txt_real}, df_do_coadapt)

  return errG, gradParametersG
end


local fAx = function(x)
  gradParametersG:zero()

  local fake = netG:forward{input_img_real, input_txt_real}
  input_img:copy(fake)
  errA = mseCriterion:forward(fake, input_img_real)
  local df_do = mseCriterion:backward(fake, input_img_real)
  netG:backward({input_img_real, input_txt_real}, df_do)

  return errA, gradParametersG
end


-- train
for epoch = 1, opt.niter do
  epoch_tm:reset()
  if epoch == 1 then
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. 0 .. '_net_G.t7', netG)
  end
  if epoch % opt.decay_every == 0 then
    optimStateG.learningRate = optimStateG.learningRate * opt.lr_decay
    optimStateD.learningRate = optimStateD.learningRate * opt.lr_decay
  end

  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
    tm:reset()

    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    optim.adam(fDx, parametersD, optimStateD)

    -- (2) Update G network: maximize log(D(G(z)))
    optim.adam(fGx, parametersG, optimStateG)

    optim.adam(fAx, parametersG, optimStateG)

    -- logging
    if ((i-1) / opt.batchSize) % opt.print_every == 0 then
      print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                .. '  Err_G: %.4f Err_Rec: %.4f Err_Adapt: %.4f Err_D: %.4f Err_R: %.4f Err_W: %.4f Err_F: %.4f Err_A: %.4f'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateG.learningRate,
              errG and errG or -1, errRec and errRec or -1,
              errAdapt and errAdapt or -1,
              errD and errD or -1, errR and errR or -1,
              errW and errW or -1, errF and errF or -1,
              errA and errA or -1))
      local fake = netG.output
      disp.image(fake:narrow(1,1,opt.batchSize), {win=opt.display_id, title=opt.name})
      disp.image(real_img, {win=opt.display_id * 3, title=opt.name})
    end
  end

  -- save checkpoints
  if epoch % opt.save_every == 0 then
    paths.mkdir(opt.checkpoint_dir)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_opt.t7', opt)
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
           epoch, opt.niter, epoch_tm:time().real))
  end
end
