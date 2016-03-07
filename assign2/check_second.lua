require 'xlua'
require 'optim'
require 'unsup'
require 'convkmeans_enh'
dofile './provider_second.lua'
c = require 'trepl.colorize'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-group', 999, 'Group number')
cmd:option('-numWindows', 100000, 'Number of windows')
cmd:option('-batchSize', 2000, 'Batch size')
cmd:option('-saveFilePrefix', 'models/second', 'Save filename prefix')
cmd:text()
opt = cmd:parse(arg or {})

saveFile = opt.saveFilePrefix .. '_' .. opt.group .. '.t7'
print('will save kernels to '..saveFile)

torch.setnumthreads(8)

print(c.blue '==> Group '.. opt.group)
provider_second = Provider_Second(opt.group)
collectgarbage()

data_dim = {4,10,10}
pSize = 5
trsize = provider_second.secondData:size()
numWindows = opt.numWindows
ngroups = 24

print(c.blue "==> find clusters")
ncentroids = 96

--Define a callback to display filters at each iteration
function dispfilters (step,c_kernels,c_counts)
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
   
   local _,varind = torch.sort(variance,1,true)
   print("Counts for iteration "..step)
   print(c_counts:reshape(1,c_counts:size(1)))
   if itorch then
       itorch.image(c_kernels:index(1,varind[{{1,math.min(96,(#varind)[1])}}]))
   end
   return c_kernels
end

kernels, counts = unsup.convkmeans(provider_second, ncentroids, data_dim[1], pSize, pSize , numWindows, 15, opt.batchSize, dispfilters, true)
resized_kernels = dispfilters(15,kernels,counts)
print('==> saving centroids to disk: ' .. saveFile)
obj = {
    resized_kernels = resized_kernels,
    kernels = kernels,
    counts = counts
}
torch.save(saveFile, obj)

