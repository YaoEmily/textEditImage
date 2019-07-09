
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'lfs'
torch.setdefaulttensortype('torch.FloatTensor')

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k
end

opt = {

  numCaption = 4,
  replicate = 1, -- if 1, then replicate averaged text features numCaption times.
  save_every = 100,
  print_every = 1,
  dataset = 'cub_fast',       -- imagenet / lsun / folder
  no_aug = 0,
  keep_img_frac = 1.0,
  interp_weight = 0,
  interp_type = 1,
  cls_weight = 0,
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
  txtSize = 1024,         -- #  of dim for raw text.
  fineSize = 64,
  nt = 128,               -- #  of dim for text features.
  nz = 50,               -- #  of dim for Z
  ngf = 128,              -- #  of gen filters in first conv layer
  ndf = 64,               -- #  of discrim filters in first conv layer
  nThreads = 1,           -- #  of data loading threads to use
  niter = 1000,             -- #  of iter at starting learning rate
  lr = 0.00002,            -- initial learning rate for adam
  lr_decay = 0.5,            -- initial learning rate for adam
  decay_every = 100,
  beta1 = 0.5,            -- momentum term of adam
  ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
  display = 1,            -- display samples while training. 0 = false
  display_id = 10,        -- display window id.
  gpu = 2,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  name = 'experiment_long',
  noise = 'normal',       -- uniform / normal
  init_g = '',
  init_d = '',
  use_cudnn = 0,

  filenames = '',
  dataset = 'cub',
  batchSize = 24,        -- number of samples to produce
  noisetype = 'normal',  -- type of noise distribution (uniform / normal).
  imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
  noisemode = 'random',  -- random / line / linefull1d / linefull
  gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
  display = 0,           -- Display image: 0 = false, 1 = true
  nz = 50,
  doc_length = 201,
  queries = 'cub_queries.txt',
  net_gen = '',
  net_txt = '',
  path = '',
  loadSize = 76,
  fineSize = 64,
  loadSize_stage2 = 140,
  fineSize_stage2 = 128,
  net_gen_stage2 = '',
  train = 0,
  stage2 = 1,
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

if opt.display == 0 then opt.display = false end

local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())



--net_gen训练好的网络
net_gen = torch.load(opt.checkpoint_dir .. '/' .. opt.net_gen)
net_txt = torch.load(opt.net_txt)

if net_txt.protos ~=nil then net_txt = net_txt.protos.enc_doc end

net_gen:evaluate()
net_txt:evaluate()

print(net_gen:size())

local img = torch.Tensor(opt.batchSize, 3, 64, 64)
local img_128 = torch.Tensor(opt.batchSize, 3, 128, 128)
img:cuda()
img_128:cuda()

-- Extract all text features.
local fea_txt = {}
local fea_img = {}
local fea_img_128 = {}
local ori_img = {}
local ori_img_128 = {}
-- Decode text for sanity check.
local raw_txt = {}
local raw_img = {}
for query_str in io.lines(opt.queries) do
  -- opt.queries = 'cub_queries.txt' 存储了测试文本 3个
  local txt = torch.zeros(1,opt.doc_length,#alphabet)
  for t = 1,opt.doc_length do
    local ch = query_str:sub(t,t)
    local ix = dict[ch]
    if ix ~= 0 and ix ~= nil then
      txt[{1,t,ix}] = 1
    end
  end
  raw_txt[#raw_txt+1] = query_str
  txt = txt:cuda()
  --print(txt:size()) --1*201*70
  fea_txt[#fea_txt+1] = net_txt:forward(txt):clone()

  real_img, real_img_128 = data:getImages()
  img_128:copy(real_img_128)
  fea_img_128[#fea_img_128+1] = real_img_128:clone()
  ori_img_128[#ori_img_128+1] = img_128:clone()
  img:copy(real_img)
  fea_img[#fea_img+1] = real_img:clone()
  ori_img[#ori_img+1] = img:clone()
end

--[[
image.save('./test1.png', real_img[1]:mul(2):add(-1))
image.save('./test2.png', real_img[5]:mul(0.5))
image.save('./test3.png', real_img[9]:mul(0.5))
image.save('./test4.png', real_img[13]:mul(0.5))
--]]

--print(fea_txt[1]:size())--1*1024
--print(fea_img[1]:size())--1*512*4*4

if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  net_gen:cuda()
  net_txt:cuda()
end

local html = '<html><body><h1>Generated Images</h1><table border="1px solid gray" style="width=100%"><tr><td><b>Caption</b></td><td><b>Image</b></td></tr>'

local images_show = torch.Tensor(48, 3, 64, 64)
local images_show_128 = torch.Tensor(48, 3, 128, 128)

for i = 1, #fea_txt do
  print(string.format('generating %d of %d', i, #fea_txt))
  local cur_fea_txt = torch.repeatTensor(fea_txt[i], opt.batchSize, 1)
  local cur_fea_img = fea_img[i]
  local cur_raw_txt = raw_txt[i]
  cur_fea_img_128 = fea_img_128[i]
  local images = net_gen:forward{cur_fea_img_128:cuda(), cur_fea_txt:cuda()}
  local visdir = string.format('results/%s', opt.dataset)
  lfs.mkdir('results')
  lfs.mkdir(visdir)
  local fname = string.format('%s/img_%d', visdir, i)
  local fname_png = fname .. '.png'
  local fname_txt = fname .. '.txt'
  local fname_png_128 = fname .. '_128.png'

  image.save('./testx.png', ori_img[i][1])
  images_show:narrow(1, 1, 8):copy(ori_img[i]:narrow(1, 1, 8))
  images_show:narrow(1, 9, 8):copy(images[1]:narrow(1, 1, 8))
  images_show:narrow(1, 17, 8):copy(ori_img[i]:narrow(1, 9, 8))
  images_show:narrow(1, 25, 8):copy(images[1]:narrow(1, 9, 8))
  images_show:narrow(1, 33, 8):copy(ori_img[i]:narrow(1, 17, 8))
  images_show:narrow(1, 41, 8):copy(images[1]:narrow(1, 17, 8))
  image.save(fname_png, image.toDisplayTensor(images_show,4,opt.batchSize/3))

  images_show_128:narrow(1, 1, 8):copy(ori_img_128[i]:narrow(1, 1, 8))
  images_show_128:narrow(1, 9, 8):copy(images[2]:narrow(1, 1, 8))
  images_show_128:narrow(1, 17, 8):copy(ori_img_128[i]:narrow(1, 9, 8))
  images_show_128:narrow(1, 25, 8):copy(images[2]:narrow(1, 9, 8))
  images_show_128:narrow(1, 33, 8):copy(ori_img_128[i]:narrow(1, 17, 8))
  images_show_128:narrow(1, 41, 8):copy(images[2]:narrow(1, 17, 8))
  image.save(fname_png_128, image.toDisplayTensor(images_show_128,4,opt.batchSize/3))

  html = html .. string.format('\n<tr><td>%s</td><td><img src="%s"></td></tr>',
                               cur_raw_txt, fname_png)
  os.execute(string.format('echo "%s" > %s', cur_raw_txt, fname_txt))
end

html = html .. '</html>'
fname_html = string.format('%s.html', opt.dataset)
os.execute(string.format('echo "%s" > %s', html, fname_html))
