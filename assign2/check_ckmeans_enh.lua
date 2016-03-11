require 'xlua'
require 'optim'
require 'unsup'
require 'ckmeans_enh'
dofile './provider_unlabel.lua'
c = require 'trepl.colorize'

torch.setnumthreads(8)

print(c.blue '==>' ..' loading data')
provider_unlabel = Provider_Unlabel()
collectgarbage()

data_dim = {3,96,96}
pSize = 3
trsize = provider_unlabel.unlabeledData:size()
numWindows = 3000000

print(c.blue "==> find clusters")
ncentroids = 64+4

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
   
   local _,varind = torch.sort(variance,1,true)
   --print("Counts for iteration "..step)
   --print(c_counts)
   if itorch then
       itorch.image(c_kernels:index(1,varind[{{1,math.min(96,(#varind)[1])}}]))
   end
   resized_kernels = c_kernels

   sfile = 'models/ckmeans'..pSize..'x'..pSize..'_'..ncentroids..'.t7'
   print('==> saving centroids to disk: ' .. sfile)
   obj = {
       resized_kernels = resized_kernels,
       kernels = c_kernels,
       counts = c_counts
   }
   torch.save(sfile, obj)
end

kernels, counts = unsup.ckmeans(provider_unlabel, ncentroids, 3, pSize, pSize , numWindows, 15, 100000, dispfilters, true)
print("==> select distinct features")
--resized_kernels = dispfilters(10,kernels,counts)


