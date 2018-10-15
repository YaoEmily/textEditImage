require 'image'
dir = require 'pl.dir'

trainLoader = {}

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
  dict[alphabet:sub(i,i)] = i
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k
end
alphabet_size = #alphabet
function decode(txt)
  local str = ''
  for w_ix = 1,txt:size(1) do
    local ch_ix = txt[w_ix]
    local ch = ivocab[ch_ix]
    if (ch  ~= nil) then
      str = str .. ch
    end
  end
  return str
end

function trainLoader:decode2(txt)
  local str = ''
  _, ch_ixs = txt:max(2)
  for w_ix = 1,txt:size(1) do
    local ch_ix = ch_ixs[{w_ix,1}]
    local ch = ivocab[ch_ix]
    if (ch ~= nil) then
      str = str .. ch
    end
  end
  return str
end

trainLoader.alphabet = alphabet
trainLoader.alphabet_size = alphabet_size
trainLoader.dict = dict
trainLoader.ivocab = ivocab
trainLoader.decode = decoder

local classnames = {}
for line in io.lines(opt.classnames) do
  classnames[#classnames + 1] = line
end

local files = {}
local trainids = {}
local size = 0
for line in io.lines(opt.trainids) do --opt.trainids = 150
  local id = tonumber(line)
  local dirpath = opt.data_root .. '/' .. classnames[id]
  cur_files = dir.getfiles(dirpath)
  files[id] = cur_files
  size = size + #cur_files
  trainids[#trainids + 1] = id
end

--------------------------------------------------------------------------------------------
local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}






local function loadImage(path)
  local input = image.load(path, 3, 'float')
  input = image.scale(input, loadSize[2], loadSize[2]) --76*76
  return input
end





-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(path)
  collectgarbage()
  --print("path =", path)
  local input = loadImage(path) -- 3*76*76
  if opt.no_aug == 1 then
    return image.scale(input, sampleSize[2], sampleSize[2])
  end

  local iW = input:size(3) -- 76
  local iH = input:size(2) -- 76

  -- do random crop
  local oW = sampleSize[2] -- 64
  local oH = sampleSize[2] -- 64
  local h1 = math.ceil(torch.uniform(1e-2, iH-oH)) -- 0.01~12 random number
  local w1 = math.ceil(torch.uniform(1e-2, iW-oW)) -- 0.01~12 random number
  local out = image.crop(input, w1, h1, w1 + oW, h1 + oH) -- 3*64*64 cut image
  assert(out:size(2) == oW)
  assert(out:size(3) == oH)
  -- do hflip with probability 0.5
  if torch.uniform() > 0.5 then out = image.hflip(out); end
  out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
  return out
end





function trainLoader:sample(quantity)
  -- quantity = 64
  if opt.replicate == 1 then
    return self:sample_repl(quantity)
  else
    return self:sample_no_repl(quantity)
  end
end




function trainLoader:single_img()
  local img = trainHook(img_file1)
  return img
end




function trainLoader:sample_no_repl(quantity)
  -- quality = 64
  local ix_batch1 = torch.Tensor(quantity)
  local ix_batch2 = torch.Tensor(quantity)
  local ix_file1 = torch.Tensor(quantity)
  local ix_file2 = torch.Tensor(quantity)

  -- Share image / text among adjacent opt.numCaption data samples.
  opt.numCaption = 1
  -- print("#trainids =", #trainids) -- 150
  for n = 1, quantity do
    local cls_ix = torch.randperm(#trainids):narrow(1,1,2) -- 打乱并取前两个
    local id1 = trainids[cls_ix[1]]
    local id2 = trainids[cls_ix[2]]
    -- print("id1 =", id1)
    -- print("id2 =", id2)
    local file_ix1 = torch.randperm(#files[id1])[1]
    local file_ix2 = torch.randperm(#files[id2])[1]
    -- print("#files[id1] =", files[id1])
    -- print("#files[id2] =", files[id2])
    -- print("file_ix1 =", file_ix1)
    -- print("file_ix2 =", file_ix2)
    ix_batch1[n] = cls_ix[1]
    ix_batch2[n] = cls_ix[2]
    ix_file1[n] = file_ix1
    ix_file2[n] = file_ix2
  end

  local data_img1 = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2]) -- real img
  local data_img2 = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2]) -- wrong img
  local data_txt1 = torch.zeros(quantity*opt.numCaption, opt.txtSize) -- real txt
  local ids = torch.zeros(quantity) -- 64个0

  local txt_batch_ix = 1
  for n = 1, quantity do
    local id1 = trainids[ix_batch1[n]] -- real img
    local id2 = trainids[ix_batch2[n]] -- wrong img
    -- print("id1 =", id1)
    -- print("id2 =", id2)

    ids[n] = id1
    local cls1_files = files[id1] -- real img
    local cls2_files = files[id2] -- wrong img

    local t7file1 = cls1_files[ix_file1[n]] -- real img
    local t7file2 = cls2_files[ix_file2[n]] -- wrong img
    -- print("t7file1 =", t7file1)
    -- print("t7file2 =", t7file2)

    local info1 = torch.load(t7file1) -- real img
    local info2 = torch.load(t7file2) -- wrong img
    -- print("info1 =", info1)
    -- print("info2 =", info2)

    local img_file1 = opt.img_dir .. '/' .. info1.img
    local img1 = trainHook(img_file1) -- real img
    local img_file2 = opt.img_dir .. '/' .. info2.img
    local img2 = trainHook(img_file2) -- wrong img

    for s = 1, opt.numCaption do
      local ix_txt1 = torch.randperm(info1.txt:size(1))[1]
      data_txt1[txt_batch_ix]:copy(info1.txt[ix_txt1])
      txt_batch_ix = txt_batch_ix + 1
    end

    data_img1[n]:copy(img1) -- real img
    data_img2[n]:copy(img2) -- wrong img
    -- print("data_img1:size() =", data_img1:size())
    -- print("data_img2:size() =", data_img2:size())
  end
  collectgarbage(); collectgarbage()
  return data_img1, data_txt1, data_img2, ids
end





function trainLoader:sample_repl(quantity)
  local numfile = math.floor(quantity / opt.numCaption)
  assert(numfile * opt.numCaption == quantity)

  local ix_batch1 = torch.Tensor(quantity)
  local ix_batch2 = torch.Tensor(quantity)
  local ix_file1 = torch.Tensor(quantity)
  local ix_file2 = torch.Tensor(quantity)
  local n = 1

  -- Share image / text among adjacent opt.numCaption data samples.
  for f = 1, numfile do
    local cls_ix = torch.randperm(#trainids):narrow(1,1,2)
    local id1 = trainids[cls_ix[1]]
    local id2 = trainids[cls_ix[2]]
    local file_ix1 = torch.randperm(#files[id1])[1]
    local file_ix2 = torch.randperm(#files[id2])[1]
    for s = 1, opt.numCaption do
      ix_batch1[n] = cls_ix[1]
      ix_batch2[n] = cls_ix[2]
      ix_file1[n] = file_ix1
      ix_file2[n] = file_ix2
      n = n + 1
    end
  end

  local data_img1 = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2]) -- real
  local data_img2 = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2]) -- wrong
  local data_txt1 = torch.zeros(quantity, opt.txtSize) -- real
  local ids = torch.zeros(quantity)

  for n = 1, quantity do
    local id1 = trainids[ix_batch1[n]]
    local id2 = trainids[ix_batch2[n]]
    ids[n] = id1
    local cls1_files = files[id1]
    local cls2_files = files[id2]

    local t7file1 = cls1_files[ix_file1[n]]
    local t7file2 = cls2_files[ix_file2[n]]
    local info1 = torch.load(t7file1)
    local info2 = torch.load(t7file2)

    local img_file1 = opt.img_dir .. '/' .. info1.img
    local img1 = trainHook(img_file1)
    local img_file2 = opt.img_dir .. '/' .. info2.img
    local img2 = trainHook(img_file2)
    local ix_txt1 = torch.randperm(info1.txt:size(1))[1]

    local txt1 = info1.txt[ix_txt1]
    data_txt1[n]:copy(txt1)
    data_img1[n]:copy(img1)
    data_img2[n]:copy(img2)
  end
  collectgarbage(); collectgarbage()
  return data_img1, data_txt1, data_img2, ids
end

function trainLoader:size()
  return size
end