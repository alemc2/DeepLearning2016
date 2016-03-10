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
cmd:option('-model', 'simple', 'Model name')
cmd:option('-firstlayer', 'models/kmeans_96.t7', 'First layer centroids')
cmd:option('-trainfirst', 999, 'The epoch at which to start training first layer')
cmd:option('-secondprefix', 'models/second', 'Second layer centroids file prefix')
cmd:option('-trainsecond', 999, 'The epoch at which to start training second layer. use 1 to always train.')
cmd:option('-bestclassifier', 'models/best_classifier.net', 'The best classifier before clustering second layer.')
cmd:option('-trainconn', 1, 'The epoch at which to start training connection layer. use 1 to always train.')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-beta1', 0.9, 'beta1 (for Adam)')
cmd:option('-beta2', 0.999, 'beta2 (for Adam)')
cmd:option('-epsilon', 1e-8, 'epsilon (for Adam)')
cmd:option('-batchSize', 64, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-patience', 100, 'minimum number of epochs to train for')
cmd:option('-epoch_step', 25, 'epoch step')
cmd:option('-improvementThreshold', 0.999, 'amount to multiply test accuracy to determine significant improvement')
cmd:option('-patienceIncrease', 2, 'amount to multiply patience by on significant improvement')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)

do -- data augmentation module
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
            local gmimage = gm.Image(input[i],'RGB','DHW')
            -- Detect edge average color to fill black regions after augmentation
            local rmean = (input[{i,1,{},1}]:sum()+input[{i,1,1,{}}]:sum()+input[{i,1,{},imwidth}]:sum()+input[{i,1,imheight,{}}]:sum())/(2*(imheight+imwidth))
            local gmean = (input[{i,2,{},1}]:sum()+input[{i,2,1,{}}]:sum()+input[{i,2,{},imwidth}]:sum()+input[{i,2,imheight,{}}]:sum())/(2*(imheight+imwidth))
            local bmean = (input[{i,3,{},1}]:sum()+input[{i,3,1,{}}]:sum()+input[{i,3,{},imwidth}]:sum()+input[{i,3,imheight,{}}]:sum())/(2*(imheight+imwidth))
            -- Rotation angle between -15 and 15 degrees
            -- This parameter with the scaling constants/some extra constants can be used to introduce shear but not doing for now as I am unable to figure exact relation to shear angle
            local theta = torch.rand(1):mul(math.pi/8):add(-math.pi/16)[1]
            -- Translation parameters +/- 2 pixels
            local tx = torch.random(-2,2)
            local ty = torch.random(-2,2)
            -- Scaling parameters 0.8-1.2 times
            local sx = (((torch.rand(1)-0.5)*0.2)+1)[1]
            local sy = (((torch.rand(1)-0.5)*0.2)+1)[1]
            -- Set backgroundand perform affine transform
            gmimage:setBackground(rmean,gmean,bmean):affineTransform(sx*math.cos(theta), sx*math.sin(theta), -sy*math.sin(theta), sy*math.cos(theta),tx,ty)
            gmw,gmh = gmimage:size()
            -- If image got bigger due to scaling/rotation we crop to keep image at same height and indicate loss of patches. Resize to original size if smaller. So down scaling isn't perfect yet.
            gmimage:crop(imwidth,imheight,(gmw-imwidth)/2,(gmh-imheight)/2):size(imwidth,imheight)
            input[i] = gmimage:toTensor('float','RGB','DHW')
         end
      end
      self.output:set(input)
      return self.output
   end
end


print '==> loading provider'
provider = torch.load('provider.t7')

print '==> construct model'

if opt.trainconn > 1 then
   connModel = torch.load(opt.bestclassifier)
   connLayer = connModel.model:get(3):get(4)

   connLayerAccGradParams = connLayer.accGradParameters
   connLayer.accGradParameters = function() end
end

firstLayer = nn.SpatialConvolution(3, 96, 13, 13, 4, 4)
firstLayer.bias:zero()
firstLayer.weight = torch.load(opt.firstlayer).resized_kernels

firstLayerAccGradParams = firstLayer.accGradParameters
firstLayer.accGradParameters = function() end

secondLayer = {}

for i=1, 24 do
   layer = nn.SpatialConvolution(4, 64, 5, 5)
   if opt.trainsecond > 1 then
      layer.bias:zero()
      layer.weight = torch.load(opt.secondprefix .. '_' .. i .. '.t7').resized_kernels
   end

   secondLayerAccGradParams = layer.accGradParameters
   layer.accGradParameters = function() end
   table.insert(secondLayer, layer)
end

model = nn.Sequential()
model:add(nn.DataAugment():float())
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
model:add(firstLayer:cuda())
model:add(dofile('models/'..opt.model..'.lua'):cuda())

criterion = nn.CrossEntropyCriterion():cuda()


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
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ADAM' then
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

   if opt.trainconn > 1 and epoch == opt.trainconn then
      print('turning on training for connection layer')
      connLayer.accGradParameters = connLayerAccGradParams
   end

   if epoch == opt.trainsecond then
      print('turning on training for second layer')
      for i = 1, 24 do
         secondLayer[i].accGradParameters = secondLayerAccGradParams
      end
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

      local inputs = provider.trainData.data:index(1,v)
      targets:copy(provider.trainData.labels:index(1,v))

      local feval = function(x)
         if x ~= parameters then parameters:copy(x) end
         gradParameters:zero()

         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

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

   train_acc = confusion.totalValid * 100

   -- save/log current net
   if epoch % 10 == 0 then
      local filename = 'models/recent.net'
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('==> saving model to '..filename)
      saveModel = model:clone()
      saveModel:remove(1)
      torch.save(filename, {model=saveModel, testData=provider.testData})
      saveModel = nil
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

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')

   local bs = 25
   for i=1,provider.valData.data:size(1),bs do
      -- disp progress
      xlua.progress(i, provider.valData.size())

      inputs = provider.valData.data:narrow(1,i,bs)
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
bestModel = nil
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
      bestModel = model:clone()
      local filename = 'models/best.net'
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('==> saving final model to '..filename)
      bestModel:remove(1)
      torch.save(filename, {model=bestModel, testData=provider.testData})
      bestModel = nil
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
