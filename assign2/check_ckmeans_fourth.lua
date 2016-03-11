require 'xlua'
require 'optim'
require 'unsup'
require 'ckmeans_enh'
dofile './provider_fourth.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-numWindows', 100000, 'Number of windows')
cmd:option('-batchSize', 10000, 'Batch size')
cmd:option('-saveFile', 'models/ckmeans_fourth256x3x3.t7', 'Save filename')
cmd:text()
opt = cmd:parse(arg or {})

print('will save kernels to '..opt.saveFile)

torch.setnumthreads(8)

provider_fourth = Provider_Fourth()
collectgarbage()

data_dim = {256,8,8}
pSize = 3
trsize = provider_fourth.fourthData:size()
numWindows = opt.numWindows

print("==> find clusters")
ncentroids = 256

--Define a callback to display filters at each iteration
function dispfilters (step,c_kernels,c_counts)
   if step==0 then
      return
   end
   local j = 0
   local variance = torch.Tensor(c_counts:size())
   for i = 1,ncentroids do
      if c_counts[i] > 0 then
         j = j + 1
         c_kernels[{j,{}}] = c_kernels[{i,{}}]
         c_counts[j] = c_counts[i]
         variance[j] = c_kernels[{j,{}}]:var()
      end
   end
   c_kernels = c_kernels[{{1,j},{}}]
   c_counts  = c_counts[{{1,j}}]
   variance  = variance[{{1,j}}]

   print("Counts for iteration "..step)
   print(c_counts:reshape(1,c_counts:size(1)))
   resized_kernels = c_kernels

   print('==> saving centroids to disk: ' .. opt.saveFile)
   obj = {
       resized_kernels = resized_kernels,
       kernels = c_kernels,
       counts = c_counts
   }
   torch.save(opt.saveFile, obj)

   collectgarbage()
end

kernels, counts = unsup.ckmeans(provider_fourth, ncentroids, data_dim[1], pSize, pSize , numWindows, 15, opt.batchSize, dispfilters, true)
--dispfilters(10,kernels,counts)
