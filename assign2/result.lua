require 'torch'   -- torch
require 'nn'      -- provides a normalization operator
require 'cunn'
require 'optim'
require 'csvigo'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-model', 'models/best.net', 'input model')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')

do -- data augmentation module blank module
   local DataAugment,parent = torch.class('nn.DataAugment', 'nn.Module')

   function DataAugment:__init()
      parent.__init(self)
      self.train = true
   end

   function DataAugment:updateOutput(input)
      self.output:set(input)
      return self.output
   end
end

print ('==> loading best model')
modelObj = torch.load(opt.model) -- TODO change for final submission

model = modelObj.model
testData = modelObj.testData

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

