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
   img_dir = '',
   keep_img_frac = 1.0,
   interp_weight = 0,
   interp_type = 1,
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

netR = nn.Sequential()
if opt.replicate == 1 then
  netR:add(nn.Reshape(opt.batchSize / opt.numCaption, opt.numCaption, opt.txtSize))
  netR:add(nn.Transpose({1,2}))
  netR:add(nn.Mean(1))
  netR:add(nn.Replicate(opt.numCaption))
  netR:add(nn.Transpose({1,2}))
  netR:add(nn.Reshape(opt.batchSize, opt.txtSize))
else
  netR:add(nn.Reshape(opt.batchSize, opt.numCaption, opt.txtSize))
  netR:add(nn.Transpose({1,2}))
  netR:add(nn.Mean(1))
end

local criterion = nn.BCECriterion()
local absCriterion = nn.AbsCriterion()
local mseCriterion = nn.MSECriterion()
local weights = torch.zeros(opt.batchSize * 3/2)
weights:narrow(1,1,opt.batchSize):fill(1)
weights:narrow(1,opt.batchSize+1,opt.batchSize/2):fill(opt.interp_weight)
local criterion_interp = nn.BCECriterion(weights)

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
local input_img_interp = torch.Tensor(opt.batchSize * 3/2, 3, opt.fineSize, opt.fineSize)
if opt.replicate == 1 then
  input_txt_real_raw = torch.Tensor(opt.batchSize, opt.txtSize)
  input_txt_wrong_raw = torch.Tensor(opt.batchSize, opt.txtSize)
else
  input_txt_real_raw = torch.Tensor(opt.batchSize * opt.numCaption, opt.txtSize)
  input_txt_wrong_raw = torch.Tensor(opt.batchSize * opt.numCaption, opt.txtSize)
end
local input_txt_real = torch.Tensor(opt.batchSize, opt.txtSize)
local input_txt_wrong = torch.Tensor(opt.batchSize, opt.txtSize)
local input_txt_interp = torch.zeros(opt.batchSize * 3/2, opt.txtSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local noise_interp = torch.Tensor(opt.batchSize * 3/2, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local label_interp = torch.Tensor(opt.batchSize * 3/2)
local errD, errG, errW, errC
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

if opt.gpu > 0 then
   input_img = input_img:cuda()
   input_img_real = input_img_real:cuda()
   input_img_wrong = input_img_wrong:cuda()
   input_img_interp = input_img_interp:cuda()
   input_txt_real = input_txt_real:cuda()
   input_txt_wrong = input_txt_wrong:cuda()
   input_txt_real_raw = input_txt_real_raw:cuda()
   input_txt_wrong_raw = input_txt_wrong_raw:cuda()
   input_txt_interp = input_txt_interp:cuda()
   noise = noise:cuda()
   noise_interp = noise_interp:cuda()
   label = label:cuda()
   label_interp = label_interp:cuda()
   netD:cuda()
   netG:cuda()
   netR:cuda()
   criterion:cuda()
   absCriterion:cuda()
   mseCriterion:cuda()
   criterion_interp:cuda()
end

if opt.use_cudnn == 1 then
  cudnn = require('cudnn')
  netD = cudnn.convert(netD, cudnn)
  netG = cudnn.convert(netG, cudnn)
  netR = cudnn.convert(netR, cudnn)
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end



local fDx = function(x)
  netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  --netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

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
  emb_txt_real = netR:forward(input_txt_real_raw)
  input_txt_real:copy(emb_txt_real)
  emb_txt_wrong = netR:forward(input_txt_wrong_raw)
  input_txt_wrong:copy(emb_txt_wrong)

  if opt.interp_type == 1 then
    -- compute (a + b)/2
    input_txt_interp:narrow(1,1,opt.batchSize):copy(input_txt_real)
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):copy(input_txt_real:narrow(1,1,opt.batchSize/2))
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):add(input_txt_real:narrow(1,opt.batchSize/2+1,opt.batchSize/2))
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):mul(0.5)
  elseif opt.interp_type == 2 then
    -- compute (a + b)/2
    input_txt_interp:narrow(1,1,opt.batchSize):copy(input_txt_real)
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):copy(input_txt_real:narrow(1,1,opt.batchSize/2))
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):add(input_txt_real:narrow(1,opt.batchSize/2+1,opt.batchSize/2))
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):mul(0.5)

    -- add extrapolation vector.
    local alpha = torch.rand(opt.batchSize/2,1):mul(2):add(-1) -- alpha ~ uniform(-1,1)
    if opt.gpu >=0 then
     alpha = alpha:float():cuda()
    end
    alpha = torch.expand(alpha,opt.batchSize/2,input_txt_interp:size(2))
    local vec = (input_txt_real:narrow(1,opt.batchSize/2+1,opt.batchSize/2) -
                input_txt_real:narrow(1,1,opt.batchSize/2)):cmul(alpha)
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):add(vec)
  end

  if opt.interp_type == 1 then
    -- compute (a + b)/2
    input_img_interp:narrow(1,1,opt.batchSize):copy(input_img)
    input_img_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):copy(input_img:narrow(1,1,opt.batchSize/2))
    input_img_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):add(input_img:narrow(1,opt.batchSize/2+1,opt.batchSize/2))
    input_img_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):mul(0.5)
  end

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
  local fake = netG:forward({input_img_wrong, input_txt_real}) -- wrong image + real text
  input_img:copy(fake)
  label:fill(fake_label)

  local output = netD:forward({input_img, input_txt_real})
  local errD_fake = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local fake_weight = 1 - opt.cls_weight
  errD_fake = errD_fake*fake_weight
  df_do:mul(fake_weight)
  netD:backward(({input_img, input_txt_real}), df_do * 0.01)

  errD = errD_real + errD_fake + errD_wrong
  errW = errD_wrong

  return errD, gradParametersD
end



-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
  netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  --netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersG:zero()

  if opt.noise == 'uniform' then -- regenerate random noise
    noise_interp:uniform(-1, 1)
  elseif opt.noise == 'normal' then
    noise_interp:normal(0, 1)
  end
  local fake = netG:forward{input_img_interp, input_txt_interp}
  input_img_interp:copy(fake)
  label_interp:fill(real_label) -- fake labels are real for generator cost

  local output = netD:forward{input_img_interp, input_txt_interp}
  errG = criterion_interp:forward(output, label_interp)
  local df_do = criterion_interp:backward(output, label_interp)
  local df_dg = netD:updateGradInput({input_img_interp, input_txt_interp}, df_do)

  netG:backward({input_img_interp, input_txt_interp}, df_dg[1])
  return errG, gradParametersG
end



local fCx = function(x)
  netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  --netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersG:zero()

  local fake = netG:forward{input_img_wrong, input_txt_real}
  input_img:copy(fake)
  local fake2 = netG:forward{input_img, input_txt_wrong}
  errC = mseCriterion:forward(fake2, input_img_wrong)
  local df_do = mseCriterion:backward(fake2, input_img_wrong)
  local df_dg = netG:updateGradInput({input_img, input_txt_wrong}, df_do)
  netG:backward({input_img_wrong, input_txt_real}, df_dg[1])

  return errC, gradParametersG

end



-- train
for epoch = 1, opt.niter do
  epoch_tm:reset()

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

    optim.adam(fCx, parametersG, optimStateG)

    -- logging
    if ((i-1) / opt.batchSize) % opt.print_every == 0 then
      print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                .. '  Err_G: %.4f  Err_D: %.4f Err_W: %.4f Err_C: %.4f'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateG.learningRate,
              errG and errG or -1, errD and errD or -1,
              errW and errW or -1, errC and errC or -1))
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
