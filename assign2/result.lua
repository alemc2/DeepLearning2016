require 'torch'   -- torch
require 'nn'      -- provides a normalization operator
require 'cunn'
require 'optim'
require 'csvigo'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-model', 'models/best.net', 'input model')
cmd:option('-testfile', 'stl-10/test.t7b', 'test file')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')

--Need to make this an independent file so copying data parser to load test data
-- parse STL-10 data from table into Tensor
function parseDataLabel(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local l = torch.ByteTensor(numSamples)
   local idx = 1
   for i = 1, #d do
      local this_d = d[i]
      for j = 1, #this_d do
    t[idx]:copy(this_d[j])
    l[idx] = i
    idx = idx + 1
      end
   end
   assert(idx == numSamples+1)
   return t, l
end

do -- data augmentation module blank module
   local DataAugment,parent = torch.class('nn.DataAugment', 'nn.Module')

   function DataAugment:__init()
      parent.__init(self)
   end

   function DataAugment:updateOutput(input)
      self.output:set(input)
      return self.output
   end
end

print ('==> loading best model')
modelObj = torch.load(opt.model) -- TODO change for final submission

model = modelObj.model
local means = modelObj.means
--Get the test data and labels
local testsize = 8000
local channel = 3
local height = 96
local width = 96
local raw_test = torch.load(opt.testfile)
testData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return testsize end
}
testData.data, testData.labels = parseDataLabel(raw_test.data, testsize, channel, height, width)

--Normalize the test data as per parameters we have
-- preprocess testSet
local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,testData:size() do
  xlua.progress(i, testData:size())
   -- rgb -> yuv
   local rgb = testData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[{1}] = normalization(yuv[{{1}}])
   testData.data[i] = yuv
end
-- normalize u globally:
testData.data:select(2,2):add(-means.mean_u)
testData.data:select(2,2):div(means.std_u)
-- normalize v globally:
testData.data:select(2,3):add(-means.mean_v)
testData.data:select(2,3):div(means.std_v)
--convert back to rgb
for i = 1,testData:size() do
   xlua.progress(i, testData:size())
   local yuv = testData.data[i]
   local rgb = image.yuv2rgb(yuv)
   testData.data[i] = rgb
end

confusion = optim.ConfusionMatrix(10)

-- predict function
function predict()
   -- local vars
   local time = sys.clock()
   local predictions = {}

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')

   local bs = 25
   for i=1,testData.data:size(1),bs do
      -- disp progress
      xlua.progress(i, testData.size())

      inputs = testData.data:narrow(1,i,bs)
      local outputs = model:forward(inputs)
      confusion:batchAdd(outputs, testData.labels:narrow(1,i,bs))

      _, maxind = torch.max(outputs, 2)
      maxind = maxind:view(bs)
      for j=0,bs-1 do
         table.insert(predictions, {i+j, maxind[j+1]})
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / testData.size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   retVal = confusion.averageValid

   -- next iteration:
   confusion:zero()

   collectgarbage()
   return retVal, predictions
end

acc, pred = predict()
print('Accuracy was ' .. acc)


-- Verify that things going into csv is correct
csvacc = testData.labels:eq(torch.Tensor(pred)[{{},2}]):float():mean()
print('Accuracy from values to be stored in csv is ' .. csvacc)

-- save csv
table.insert(pred,1,{"Id","Prediction"})
csvigo.save{ path='predictions.csv', data=pred }

