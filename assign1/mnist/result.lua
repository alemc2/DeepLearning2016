----------------------------------------------------------------
-- This script is used to generate the test resullts from the 
-- stored model and to generate a csv output file
----------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'
require 'csvigo'

-----------------------------------------------------------------

print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST Result generation')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
cmd:option('-type', 'cuda', 'type: double | float | cuda') -- XXX: use double for final submission
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
-- data:
cmd:option('-size', 'full', 'how many samples do we load: small | full')
cmd:option('-model', 'results/finalmodel.net', 'relative path of model file: results/finalmodel.net')
cmd:option('-testset', 'mnist.t7/test_32x32.t7' , 'relative path of test data set : mnist.t7/test_32x32.t7') 
cmd:option('-output', 'predictions.csv', 'relative path where you want to stor predicitions csv file : predictions.csv')
--------------------------------------------------------------------
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
-- Path setup
test_file = paths.concat(opt.testset)
model_file = paths.concat(opt.model)
out_file = paths.concat(opt.output)


----------------------------------------------------------------------
-- test size

if opt.size == 'full' then
   print '==> using regular, full testing data'
   tesize = 10000
elseif opt.size == 'small' then
   print '==> using reduced testing data, for fast experiments'
   tesize = 2000
end
-- Load test data
loaded = torch.load(test_file,'ascii')
test_data = loaded.data[{ {1,tesize}, {}, {}, {} }]
test_labels = loaded.labels[{ {1,tesize} }]
testData = {
   data = test_data,
   labels = test_labels,
   size = function() return tesize end
}
-- Load model
loaded = torch.load(model_file)
model = loaded.model
mean = loaded.mean
std = loaded.std

----------------------------------------------------------------------
print '==> preprocessing data'

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch, 
-- in general by doing: dst = src:type('torch.TypeTensor'), 
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

testData.data = testData.data:float()
-- Normalize test data, using the training means/stds
testData.data[{ {},1,{},{} }]:add(-mean)
testData.data[{ {},1,{},{} }]:div(std)

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   local predictions = {}
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
      -- Simple dimension checks
      assert(pred:dim()==1)
      -- Take confident prediction
      _,maxind = pred:max(1)
      table.insert(predictions, {t,maxind[1]})
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end

   retVal = confusion.averageValid

   -- next iteration:
   confusion:zero()

   return retVal,predictions
end
----------------------------------------------------------------------
print '==> testing!'

-- classes
classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)


-- Log results to files
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

----------------------------------------------------------------------
acc,pred = test()

print('Accuracy was ' .. acc)

----------------------------------------------------------------------
-- Verify that things going into csv is correct
csvacc = testData.labels:eq(torch.Tensor(pred)[{{},{2}}]:byte()):float():mean()
print('Accuracy from values to be stored in csv is ' .. csvacc)

-- save csv
table.insert(pred,1,{"Id","Prediction"})
csvigo.save{ path=opt.output, data=pred }
