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
    ngf = 128,              -- #  of gen filters in first conv layer
    ndf = 64,               -- #  of discrim filters in first conv layer
    fineSize = 64,
    loadSize = 76,
    batchSize = 64,
    numCaption = 4,
    nThreads = 4,           -- #  of data loading threads to use
    dataset = 'cub',       -- imagenet / lsun / folder
    save_every = 10,
    lr = 0.0001,            -- initial learning rate for adam
    lr_decay = 0.5,            -- initial learning rate for adam
    decay_every = 100,
    beta1 = 0.5,            -- momentum term of adam
    ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
    niter = 300,             -- #  of iter at starting learning rate
    print_every = 4,
    nThreads = 4,
    replicate = 1,
    display = 1,            -- display samples while training. 0 = false
    display_id = 10,        -- display window id.
    name = 'experiment_long',

    interp_type = 1,
    cls_weight = 0.5,

    data_root = '/home/xhy/code/icml2016/dataset_cub/cub_icml',
    classnames = '/home/xhy/code/icml2016/dataset_cub/cub_icml/allclasses.txt',
    trainids = '/home/xhy/code/icml2016/dataset_cub/cub_icml/trainvalids.txt',
    img_dir = '/home/xhy/code/icml2016/dataset_cub/CUB_200_2011/images',
    checkpoint_dir = '/home/xhy/code/icml2016/checkpoints',
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
local nc = 3
local real_label = 1
local fake_label = 0

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local criterion_img = nn.MSECriterion()
criterion_img.sizeAverage = true
local criterion = nn.BCECriterion()
criterion.sizeAverage = false

local input_img = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
if opt.replicate == 1 then
  input_txt_raw = torch.Tensor(opt.batchSize, opt.txtSize)
else
  input_txt_raw = torch.Tensor(opt.batchSize * opt.numCaption, opt.txtSize)
end
local input_txt = torch.Tensor(opt.batchSize, opt.txtSize)
local input_txt_interp = torch.zeros(opt.batchSize * 3/2, opt.txtSize)
local label = torch.Tensor(opt.batchSize)

local output_real = torch.Tensor(1, 64)
local output_wrong = torch.Tensor(1, 64)

local errD, errG, errW
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

optimStateD = {
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

if opt.init_d == '' then
    -- input image and text: image 64*3*64*64 text64*1024*1*1

    convD = nn.Sequential()
    -- input is (nc) x 64 x 64
    convD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
    convD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 32 x 32
    convD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    convD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 16 x 16
    convD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    convD:add(SpatialBatchNormalization(ndf * 4))
    -- state size: (ndf*4) x 8 x 8
    convD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    convD:add(SpatialBatchNormalization(ndf * 8))

    -- state size: (ndf*8) x 4 x 4
    local conc = nn.ConcatTable()
    local conv = nn.Sequential()
    conv:add(SpatialConvolution(ndf * 8, ndf * 2, 1, 1, 1, 1, 0, 0))
    conv:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    conv:add(SpatialConvolution(ndf * 2, ndf * 2, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ndf * 2))
    conv:add(nn.LeakyReLU(0.2, true))
    conv:add(SpatialConvolution(ndf * 2, ndf * 8, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ndf * 8))
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
    netD = nn.Sequential()
    pt = nn.ParallelTable()
    pt:add(convD)
    pt:add(fcD)
    netD:add(pt)
    netD:add(nn.JoinTable(2))
    -- state size: (ndf*8 + 128) x 4 x 4
    netD:add(SpatialConvolution(ndf * 8 + opt.nt, ndf * 8, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
    netD:add(nn.Sigmoid())
    -- state size: 1 x 1 x 1
    netD:add(nn.View(1):setNumInputDims(3))
    -- state size: 1

    netD:apply(weights_init)
else
    netD = torch.load(opt.init_d)
end

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

if opt.gpu > 0 then
    input_img = input_img:cuda()
    input_txt_raw = input_txt_raw:cuda()
    input_txt = input_txt:cuda()
    input_txt_interp = input_txt_interp:cuda()
    label = label:cuda()
    netD = netD:cuda()
    netR = netR:cuda()
    criterion_img = criterion_img:cuda()
    criterion = criterion:cuda()
end

local parametersD, gradParametersD = netD:getParameters()

local preNetD = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

    gradParametersD:zero()

    -- train with real
    data_tm:reset(); data_tm:resume()
    real_img, real_txt, wrong_img, _ = data:getBatch()
    data_tm:stop()

    input_img:copy(real_img)
    input_txt_raw:copy(real_txt)
    -- average adjacent text features in batch dimension.
    emb_txt = netR:forward(input_txt_raw)
    input_txt:copy(emb_txt)

    if opt.interp_type == 1 then
        -- compute (a + b)/2
        input_txt_interp:narrow(1,1,opt.batchSize):copy(input_txt)
        input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):copy(input_txt:narrow(1,1,opt.batchSize/2))
        input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):add(input_txt:narrow(1,opt.batchSize/2+1,opt.batchSize/2))
        input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):mul(0.5)
    elseif opt.interp_type == 2 then
        -- compute (a + b)/2
        input_txt_interp:narrow(1,1,opt.batchSize):copy(input_txt)
        input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):copy(input_txt:narrow(1,1,opt.batchSize/2))
        input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):add(input_txt:narrow(1,opt.batchSize/2+1,opt.batchSize/2))
        input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):mul(0.5)

        -- add extrapolation vector.
        local alpha = torch.rand(opt.batchSize/2,1):mul(2):add(-1) -- alpha ~ uniform(-1,1)
        if opt.gpu >=0 then
            alpha = alpha:float():cuda()
        end
        alpha = torch.expand(alpha,opt.batchSize/2,input_txt_interp:size(2))
        local vec = (input_txt:narrow(1,opt.batchSize/2+1,opt.batchSize/2) -
            input_txt:narrow(1,1,opt.batchSize/2)):cmul(alpha)
        input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):add(vec)
    end
    label:fill(real_label)

    local output = netD:forward{input_img, input_txt}
    print('true:')
    print(output:narrow(1, 1, 1))
    output_real:copy(output)
    local errD_real = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward({input_img, input_txt}, df_do)

    -- train with wrong
    errD_wrong = 0
    if opt.cls_weight > 0 then
        -- train with wrong
        input_img:copy(wrong_img)
        label:fill(fake_label)

        local output = netD:forward{input_img, input_txt}
        print('wrong:')
        print(output:narrow(1, 1, 1))
        output_wrong:copy(output)
        errD_wrong = opt.cls_weight*criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        df_do:mul(opt.cls_weight)
        netD:backward({input_img, input_txt}, df_do)
    end

    errD = errD_real + errD_wrong
    errW = errD_wrong

    return errD, gradParametersD
end

-- train
for epoch = 1, opt.niter do
    epoch_tm:reset()

    if epoch % opt.decay_every == 0 then
        optimStateD.learningRate = optimStateD.learningRate * opt.lr_decay
    end

    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        tm:reset()

        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        optim.adam(preNetD, parametersD, optimStateD)
        -- logging
        if ((i-1) / opt.batchSize) % opt.print_every == 0 then
            print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                .. '  Err_D: %.4f'):format(
                epoch,
                ((i-1) / opt.batchSize),
                math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                tm:time().real,
                data_tm:time().real,
                optimStateD.learningRate,
                errD and errD or -1))
            --print(fake:size()) -- 64*1
            --disp.image(fake:narrow(1,1,opt.batchSize), {win=opt.display_id, title=opt.name})
            --disp.image(real_img, {win=opt.display_id * 3, title=opt.name})
        end
    end

    -- save checkpoints
    if epoch % opt.save_every == 0 then
        paths.mkdir(opt.checkpoint_dir)
        torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD)
        print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
    end
end
