require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'cunn'
require 'optim'
gm = require 'graphicsmagick'

dofile 'provider.lua'

print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-optimization', 'ADAM', 'optimization method: SGD | ADAM')
cmd:option('-model', 'outer', 'Model name')
cmd:option('-firstlayer', 'models/ckmeans3x3b_64.t7', 'First layer centroids')
cmd:option('-trainfirst', 8, 'The epoch at which to start training first layer')
cmd:option('-secondlayer', 'models/ckmeans_second64x3x3.t7', 'Second layer centroids')
cmd:option('-trainsecond', 8, 'The epoch at which to start training second layer')
cmd:option('-thirdlayer', 'models/ckmeans_third128x3x3.t7', 'Third layer centroids')
cmd:option('-trainthird', 8, 'The epoch at which to start training third layer')
cmd:option('-fourthlayer', 'models/ckmeans_fourth256x3x3.t7', 'Fourth layer centroids')
cmd:option('-trainfourth', 8, 'The epoch at which to start training fourth layer')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-learningRateDecay', 1e-7, 'learning rate decay (SGD)')
cmd:option('-notest', 20, 'Number of epochs to skip testing for')
cmd:option('-beta1', 0.9, 'beta1 (for Adam)')
cmd:option('-beta2', 0.999, 'beta2 (for Adam)')
cmd:option('-epsilon', 1e-8, 'epsilon (for Adam)')
cmd:option('-batchSize', 64, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0.0005, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
cmd:option('-patience', 300, 'minimum number of epochs to train for')
cmd:option('-epoch_step', 50, 'epoch step')
cmd:option('-improvementThreshold', 0.999, 'amount to multiply test accuracy to determine significant improvement')
cmd:option('-patienceIncrease', 1.5, 'amount to multiply patience by on significant improvement')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)

do -- data augmentation module assume rgb for now
   function vignette(img,darken_factor) --darken_factor roughly indicates how fast it fades. Higher = more fading
       local nch = img:size(1)
       local imh = img:size(2)
       local imw = img:size(3)
       local maxdist = math.sqrt(imh^2+imw^2)/2
       for i = 1,imh do
           for j = 1,imw do
               local dist = math.sqrt((i-imh/2)^2 + (j-imw/2)^2)
               img[{{},i,j}]:mul(math.max(0,1-darken_factor*dist/maxdist))
           end
       end
   end

   -- Contrast normalization based on hsv image data from http://papers.nips.cc/paper/5548-discriminative-unsupervised-feature-learning-with-convolutional-neural-networks.pdf
   function contrastaugment_sv(img,s_factors,v_factors)
       local hsv_img = image.rgb2hsv(img)
       hsv_img[2]:pow(s_factors.s_pow):mul(s_factors.s_mul):add(s_factors.s_add)
       hsv_img[3]:pow(v_factors.v_pow):mul(v_factors.v_mul):add(v_factors.v_add)
       --Do inpace replace
       img:set(image.hsv2rgb(hsv_img))
   end

   local DataAugment,parent = torch.class('nn.DataAugment', 'nn.Module')

   function DataAugment:__init()
      parent.__init(self)
      self.train = true
   end

   function DataAugment:updateOutput(input)
      if self.train then
         local bs = input:size(1)
         local imheight = input:size(3)
         local imwidth = input:size(4)
         local flip_mask = torch.randperm(bs):le(bs/2)
         for i=1,bs do
            if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
            --Before converting to gmimage scale range to 0-1 using minmax.. Will be rescaled back.
            local im_min = input[i]:min()
            local im_max = input[i]:max()
            input[i] = image.minmax{tensor=input[i],min=im_min,max=im_max}
            -- Perform color augmentation here as per http://arxiv.org/vc/arxiv/papers/1501/1501.02876v1.pdf
            local rshift = torch.uniform(-20/255,20/255)
            local gshift = torch.uniform(-20/255,20/255)
            local bshift = torch.uniform(-20/255,20/255)
            input[i][1]:add(rshift)
            input[i][2]:add(gshift)
            input[i][3]:add(bshift)
            -- Perform contrast augmentation
            local s_factors = {s_pow=torch.uniform(0.25,2),s_mul=torch.uniform(0.7,1.4),s_add=torch.uniform(-0.1,0.1)}
            local v_factors = {v_pow=torch.uniform(0.25,2),v_mul=torch.uniform(0.7,1.4),v_add=torch.uniform(-0.1,0.1)}
            contrastaugment_sv(input[i],s_factors,v_factors)
            -- Use gmimage to do affine transforms
            local gmimage = gm.Image(input[i],'RGB','DHW')
            -- Detect edge average color to fill black regions after augmentation
            local rmean = (input[{i,1,{},1}]:sum()+input[{i,1,1,{}}]:sum()+input[{i,1,{},imwidth}]:sum()+input[{i,1,imheight,{}}]:sum())/(2*(imheight+imwidth))
            local gmean = (input[{i,2,{},1}]:sum()+input[{i,2,1,{}}]:sum()+input[{i,2,{},imwidth}]:sum()+input[{i,2,imheight,{}}]:sum())/(2*(imheight+imwidth))
            local bmean = (input[{i,3,{},1}]:sum()+input[{i,3,1,{}}]:sum()+input[{i,3,{},imwidth}]:sum()+input[{i,3,imheight,{}}]:sum())/(2*(imheight+imwidth))
            -- Rotation angle between -15 and 15 degrees
            -- This parameter with the scaling constants/some extra constants can be used to introduce shear but not doing for now as I am unable to figure exact relation to shear angle
            local theta = torch.rand(1):mul(math.pi/8):add(-math.pi/16)[1]
            -- Translation parameters +/- 10 pixels
            local tx = torch.random(-10,10)
            local ty = torch.random(-10,10)
            -- Scaling parameters 0.8-1.2 times. Also does strech when aspect ratio is lost
            local sx = (((torch.rand(1)-0.5)*0.2)+1)[1]
            local sy = (((torch.rand(1)-0.5)*0.2)+1)[1]
            -- Set backgroundand perform affine transform
            gmimage:setBackground(rmean,gmean,bmean):affineTransform(sx*math.cos(theta), sx*math.sin(theta), -sy*math.sin(theta), sy*math.cos(theta),tx,ty)
            gmw,gmh = gmimage:size()
            -- If image got bigger due to scaling/rotation we crop to keep image at same height and indicate loss of patches. Fill area if smaller.
            -- Also have resize at end to avoid any discrepancy between odd even even sizes when adding border to fill area.
            gmimage:crop(imwidth,imheight,(gmw-imwidth)/2,(gmh-imheight)/2):addBorder((imwidth-gmw)/2,(imheight-gmh)/2,rmean,gmean,bmean):size(imwidth,imheight)
            input[i] = gmimage:toTensor('float','RGB','DHW')
            --Once affine transform done do vignette.. No need to assign, does inplace
            --vignette(input[i],torch.uniform(0,1.2))
            -- Rescale range back to original
            input[i]:mul(im_max-im_min):add(im_min)
         end
      end
      self.output:set(input)
      return self.output
   end
end


print '==> loading provider'
provider = torch.load('provider.t7')

print '==> construct model'

firstLayer = nn.SpatialConvolution(3, 64, 3,3, 1,1, 1,1)
firstLayer.bias:zero()
firstLayer.weight = torch.load(opt.firstlayer).resized_kernels

firstLayerAccGradParams = firstLayer.accGradParameters
firstLayer.accGradParameters = function() end

secondLayer = nn.SpatialConvolution(64, 128, 3,3, 1,1, 1,1)
secondLayer.bias:zero()
secondLayer.weight = torch.load(opt.secondlayer).resized_kernels

secondLayerAccGradParams = secondLayer.accGradParameters
secondLayer.accGradParameters = function() end

thirdLayer = nn.SpatialConvolution(128, 256, 3,3, 1,1, 1,1)
thirdLayer.bias:zero()
thirdLayer.weight = torch.load(opt.thirdlayer).resized_kernels

thirdLayerAccGradParams = thirdLayer.accGradParameters
thirdLayer.accGradParameters = function() end

fourthLayer = nn.SpatialConvolution(256, 256, 3,3, 1,1, 1,1)
fourthLayer.bias:zero()
fourthLayer.weight = torch.load(opt.fourthlayer).resized_kernels

fourthLayerAccGradParams = fourthLayer.accGradParameters
fourthLayer.accGradParameters = function() end

model = nn.Sequential()
model:add(nn.DataAugment():float())
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
model:add(dofile('models/'..opt.model..'.lua'):cuda())

criterion = nn.ClassNLLCriterion():cuda()


print '==> here is the model:'
print(model)


-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(10)

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
parameters,gradParameters = model:getParameters()


print '==> configuring optimizer'

if opt.optimization == 'SGD' then
   print('using SGD')
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = opt.learningRateDecay
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ADAM' then
   print('using ADAM')
   optimState = {
      learningRate = opt.learningRate,
      beta1 = opt.beta1,
      beta2 = opt.beta2,
      epsilon = opt.epsilon
   }
   optimMethod = optim.adam

else
   error('unknown optimization method')
end


print '==> defining training procedure'

function train()
   model:training()
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   if epoch == opt.trainfirst then
      print('turning on training for first layer')
      firstLayer.accGradParameters = firstLayerAccGradParams
   end

   if epoch == opt.trainsecond then
      print('turning on training for second layer')
      secondLayer.accGradParameters = secondLayerAccGradParams
   end

   if epoch == opt.trainthird then
      print('turning on training for third layer')
      thirdLayer.accGradParameters = thirdLayerAccGradParams
   end

   if epoch == opt.trainfourth then
      print('turning on training for fourth layer')
      fourthLayer.accGradParameters = fourthLayerAccGradParams
   end

   -- drop learning rate every "epoch_step" epochs
   if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end

   print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

   local targets = torch.CudaTensor(opt.batchSize)
   local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
   -- remove last element so that all the batches have equal size
   indices[#indices] = nil

   local tic = torch.tic()
   for t,v in ipairs(indices) do
      xlua.progress(t, #indices)

      local inputs = provider.trainData.data:index(1,v):clone() -- XXX cloning because augmentation screws stuff up
      targets:copy(provider.trainData.labels:index(1,v))

      local feval = function(x)
         if x ~= parameters then parameters:copy(x) end
         gradParameters:zero()

         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)
         --print(torch.sum(firstLayer.weight))

         confusion:batchAdd(outputs, targets)

         return f,gradParameters
      end
      optimMethod(feval, parameters, optimState)
      collectgarbage()
   end

   -- time taken
   time = sys.clock() - time
   time = time / provider.trainData.size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   confusion:updateValids()

   -- print confusion matrix
   print(confusion)

   print(('Train accuracy: '..'%.2f'..' %%\t time: %.2f s'):format(
         confusion.totalValid * 100, torch.toc(tic)))

   -- save/log current net
   if epoch % 10 == 0 then
      local filename = 'models/recent.net'
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('==> saving model to '..filename)
      torch.save(filename, {model=model, means={mean_u = provider.mean_u, std_u  = provider.std_u, mean_v = provider.mean_v, std_v  = provider.std_v}})
   end

   confusion:zero()
   epoch = epoch + 1
   collectgarbage()
end


print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- spend at least some epochs not testing
   if epoch <= opt.notest+1 then
      return 0.0
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')

   local bs = 25
   for i=1,provider.valData.data:size(1),bs do
      -- disp progress
      xlua.progress(i, provider.valData.size())

      local inputs = provider.valData.data:narrow(1,i,bs)
      local outputs = model:forward(inputs)
      confusion:batchAdd(outputs, provider.valData.labels:narrow(1,i,bs))
   end

   -- timing
   time = sys.clock() - time
   time = time / provider.valData.size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   retVal = confusion.averageValid

   -- next iteration:
   confusion:zero()

   collectgarbage()
   return retVal
end


print '==> training!'

bestAcc = 0.0
patience = opt.patience
improvementThreshold = opt.improvementThreshold
patienceIncrease = opt.patienceIncrease

while true do
   train()
   acc = test()
   print(acc)

   if acc > bestAcc then
      print '==> found new best model!'
      local old_p = patience
      if acc * improvementThreshold > bestAcc then
         patience = math.max(patience, patienceIncrease * (epoch-1))
         if patience ~= old_p then
            print('==> increasing patience from ' .. old_p .. ' to ' .. patience)
         end
      else
         print '==> not a significant improvement'
         if patience <= epoch then
            patience = patience + 2
            print('==> increasing patience from ' .. old_p .. ' to ' .. patience)
         end
      end

      bestAcc = acc
      local filename = 'models/best.net'
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('==> saving final model to '..filename)
      torch.save(filename, {model=model, means={mean_u = provider.mean_u, std_u  = provider.std_u, mean_v = provider.mean_v, std_v  = provider.std_v}})
   end

   print('Best model accuracy is ' .. bestAcc)

   if patience <= (epoch-1) then
      print '==> out of patience'
      break
   else
      print('==> patience: ' .. patience)
   end
   collectgarbage()
end
