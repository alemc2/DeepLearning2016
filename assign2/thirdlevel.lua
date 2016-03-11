require 'torch'   -- torch
require 'nn'      -- provides a normalization operator
require 'cunn'

dofile './provider.lua'
dofile './provider_unlabel.lua'
dofile './augmentdummy.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-modelfile', 'models/bestSecond.net', 'Model file name')
cmd:option('-batchSize', 150, 'batch size')
cmd:option('-numgroups', 128, 'Number of groups')
cmd:option('-mapsize', 8, 'Output feature map size')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')

print ('==> loading data')
provider = torch.load('provider.t7')
provider_un = Provider_Unlabel()

print ('==> loading best model')
model = torch.load(opt.modelfile).model

print ('==> before changing model:')
print (model)

print ('\n==> after changing model:')
size = model:get(3):size()
removeLayers = 19
for i=size,size-removeLayers+1,-1 do
   model:get(3):remove(i)
end
print (model)
collectgarbage()

--model = nil

function generate()
   local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
   local time = sys.clock()
   model:evaluate()

   print('==> generating intermediate data')

   local finalout = torch.Tensor(provider_un.unlabeledData.size(), opt.numgroups, opt.mapsize, opt.mapsize)

   local bs = opt.batchSize
   for i=1,provider_un.unlabeledData.size(),bs do
      -- disp progress
      xlua.progress(i, provider_un.unlabeledData.size())

      local lasti = math.min(i+bs-1, provider_un.unlabeledData.data:size(1))
      local inputs = provider_un.unlabeledData.data[{{i,lasti}}]:float()

      -- preprocess valSet
      for i = 1,inputs:size(1) do
        -- rgb -> yuv
        local rgb = inputs[i]
        local yuv = image.rgb2yuv(rgb)
        -- normalize y locally:
        yuv[{1}] = normalization(yuv[{{1}}])
        inputs[i] = yuv
      end
      -- normalize u globally:
      inputs:select(2,2):add(-provider.mean_u)
      inputs:select(2,2):div(provider.std_u)
      -- normalize v globally:
      inputs:select(2,3):add(-provider.mean_v)
      inputs:select(2,3):div(provider.std_v)

      for i = 1,inputs:size(1) do
        local yuv = inputs[i]
        local rgb = image.yuv2rgb(yuv)
        inputs[i] = rgb
      end

      local outputs = model:forward(inputs):float()
      finalout[{{i,lasti}}] = outputs
      outputs = nil
      collectgarbage()
   end

   -- timing
   time = sys.clock() - time
   time = time / provider_un.unlabeledData.size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   print('==> saving data')
   torch.save('stl-10/thirdlevel.t7', finalout)
end

generate()

