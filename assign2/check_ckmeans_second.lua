require 'xlua'
require 'optim'
require 'unsup'
require 'ckmeans_enh'
dofile './provider_second.lua'
c = require 'trepl.colorize'

torch.setnumthreads(8)

print(c.blue '==>' ..' loading data')
provider_second = Provider_Second()
collectgarbage()

data_dim = {4,10,10}
pSize = 5
trsize = provider_second.secondData:size()
numWindows = 50000
ngroups = 24

print(c.blue "==> find clusters")
ncentroids = 96

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
   print("Counts for iteration "..step)
   print(c_counts:reshape(1,c_counts:size(1)))
   resized_kernels = c_kernels
end

for i=1,ngroups do
   print ('==> Group '.. i)
   provider_second.group = i
   kernels, counts = unsup.ckmeans(provider_second, ncentroids, data_dim[1], pSize, pSize , numWindows, 15, 10000, dispfilters, true)
   resized_kernels = dispfilters(10,kernels,counts)
end

