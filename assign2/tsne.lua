require 'torch'   -- torch
require 'nn'      -- provides a normalization operator
require 'cunn'
m = require 'manifold';

dofile './provider.lua'
dofile './augmentdummy.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-modelfile', 'models/best.net', 'Model file name')
cmd:option('-batchSize', 150, 'batch size')
cmd:option('-featSize', 256, 'Feature size')
cmd:option('-thumbnail', 40, 'Thumbnail size')
cmd:option('-saveImage', 'tsne.png', 'Image file')
cmd:text()
opt = cmd:parse(arg or {})

do
   local RescaleModule,parent = torch.class('nn.RescaleModule', 'nn.Module')

   function RescaleModule:__init(scale_factor)
      parent.__init(self)
      self.scale_factor = scale_factor;
   end

   function RescaleModule:updateOutput(input)
      local bs = input:size(1)
      local nch = input:size(2)
      local imheight = input:size(3)
      local imwidth = input:size(4)
      self.output = torch.CudaTensor(bs,nch,imheight*self.scale_factor,imwidth*self.scale_factor)
      for i=1,bs do
         self.output[i]:copy(image.scale(input[i]:float(),imwidth*self.scale_factor,imheight*self.scale_factor))
      end
      return self.output
   end

   function RescaleModule:updateGradInput(input, gradOutput)
      --assert(input:nElement() == gradOutput:nElement())

      self.gradInput = gradOutput:resizeAs(input)
      return self.gradInput
   end
end

torch.setdefaulttensortype('torch.FloatTensor')

print ('==> loading data')
provider = torch.load('provider.t7')

dataset = provider.trainData
--dataset.size = function() return 500 end
--dataset.data = dataset.data[{{1,dataset.size()}}]
--dataset.labels = dataset.labels[{{1,dataset.size()}}]

raw_train = torch.load('stl-10/train.t7b')
raws = parseDataLabel(raw_train.data, 4000, 3, 96, 96)
raws = raws[{{1,dataset.size()}}]

print ('==> loading best model')
model = torch.load(opt.modelfile).model:get(3):get(1):get(1):get(2);
collectgarbage()

--print ('==> before changing model:')
--print (model)

--print ('\n==> after changing model:')
size = model:get(30):size()
removeLayers = 3
for i=size,size-removeLayers+1,-1 do
   model:get(30):remove(i)
end
--print (model)
collectgarbage()

function generate()
   local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
   local time = sys.clock()
   model:evaluate()

   print('==> generating intermediate data')

   local finalout = torch.DoubleTensor(dataset.size(), opt.featSize)

   local bs = opt.batchSize
   for i=1,dataset.size(),bs do
      -- disp progress
      xlua.progress(i, dataset.size())

      local lasti = math.min(i+bs-1, dataset.data:size(1))
      local inputs = dataset.data[{{i,lasti}}]:clone():float()

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

      local outputs = model:forward(inputs:cuda()):double()
      finalout[{{i,lasti}}] = outputs
      outputs = nil
      collectgarbage()
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset.size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   print('==> saving data')
   return finalout, dataset.labels
end

x, labels = generate()

torch.setdefaulttensortype('torch.DoubleTensor')

print(x:size())
print(labels:size())

imgs = torch.Tensor(dataset.data:size(1), 3, opt.thumbnail, opt.thumbnail)
for i = 1,dataset.size() do
    imgs[i] = image.scale(raws[i],opt.thumbnail,opt.thumbnail)
end

opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta=0.5}
mapped_x1 = m.embedding.tsne(x, opts)

im_size = 4096
map_im = m.draw_image_map(mapped_x1, imgs, im_size, 0, true)

image.save(opt.saveImage, map_im:byte())
