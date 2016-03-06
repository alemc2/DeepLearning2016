require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'cunn'
require 'optim'

dofile 'provider.lua'

print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-optimization', 'ADAM', 'optimization method: SGD | ADAM')
cmd:option('-firstlayer', 'models/kmeans_96.t7', 'First layer centroids')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-beta1', 0.9, 'beta1 (for Adam)')
cmd:option('-beta2', 0.999, 'beta2 (for Adam)')
cmd:option('-epsilon', 1e-8, 'epsilon (for Adam)')
cmd:option('-batchSize', 16, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-patience', 20, 'minimum number of epochs to train for')
cmd:option('-improvementThreshold', 0.999, 'amount to multiply test accuracy to determine significant improvement')
cmd:option('-patienceIncrease', 2, 'amount to multiply patience by on significant improvement')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)

print '==> loading provider'
provider = Provider()

print '==> define parameters'

-- 10-class problem
noutputs = 10

-- input dimensions
nfeats = 3
width = 96
height = 96
ninputs = nfeats*width*height

-- hidden units, filter sizes (for ConvNet only):
nstates = 96
filtsize = 7
poolsize = 2

print '==> construct model'

model = nn.Sequential()

firstLayer = nn.SpatialConvolution(nfeats, nstates, filtsize, filtsize, 2, 2)
firstLayer.bias:zero()
firstLayer.weight = torch.load(opt.firstlayer).resized_kernels

--model:add(firstLayer) -- XXX generalize
-- TODO: do we need batch normalization?
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 3 : standard 2-layer neural network
model:add(nn.View(nstates*22*22))
--model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates*22*22, noutputs))

criterion = nn.CrossEntropyCriterion()

firstLayer:cuda() -- XXX generalize
model:cuda()
criterion:cuda()


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

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(provider.trainData.size())

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,provider.trainData.size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, provider.trainData.size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,provider.trainData.size()) do
         -- load new sample
         local input = provider.trainData.data[shuffle[i]] -- will need to clone this when applying transformations
         local target = provider.trainData.labels[shuffle[i]]
         input = input:cuda()
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local first_output = firstLayer:forward(inputs[i]) -- XXX generalize?
                          local output = model:forward(first_output)
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          local bp = model:backward(first_output, df_do)
                          if epoch > 0 then -- XXX > 0 ?
                             firstLayer:backward(inputs[i], bp) -- FIXME this doesn't work!
                             --print(torch.sum(firstLayer.weight))
                          end

                          -- update confusion
                          confusion:add(output, targets[i])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)
      collectgarbage()
   end

   -- time taken
   time = sys.clock() - time
   time = time / provider.trainData.size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- save/log current net
   local filename = 'models/recent.net'
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
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
   for t = 1,provider.valData.size() do
      -- disp progress
      xlua.progress(t, provider.valData.size())

      -- get new sample
      local input = provider.valData.data[t]
      input = input:cuda()
      local target = provider.valData.labels[t]

      -- test sample
      local pred_first = firstLayer:forward(input) -- XXX generalize
      local pred = model:forward(pred_first)
      confusion:add(pred, target)
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
bestModel = model
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
      torch.save(filename, {model=bestModel, firstlayer=firstLayer:clone()})
   end

   print('Best model accuracy is ' .. bestAcc)

   if patience <= (epoch-1) then
      print '==> out of patience'
      break
   else
      print('==> patience: ' .. patience)
   end
end
