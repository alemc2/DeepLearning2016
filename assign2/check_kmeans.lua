require 'xlua'
require 'optim'
require 'unsup'
dofile './provider.lua'
local c = require 'trepl.colorize'

torch.setnumthreads(8)

print(c.blue '==>' ..' loading data')
--provider = torch.load 'provider.t7'
--provider.trainData.data = provider.trainData.data:float()
--provider.valData.data = provider.valData.data:float()
raw_unlabel = torch.load('stl-10/extra.t7b')
unlabeled_data,_ = parseDataLabel(raw_unlabel.data,100000,3,96,96)
raw_unlabel = nil
collectgarbage()
--for i = 1,unlabeled_data:size(1) do
--    --Convert this to YUV as we do the same for labeled data but need to investigate if normalization needed
--    unlabeled_data[i] = image.rgb2yuv(unlabeled_data[i])
--end

print(c.blue "==> extract windows")
data_dim = {3,96,96}
kSize = 11
trsize = 4000
numPatches = 300000*64
patches = torch.zeros(numPatches, data_dim[1], kSize, kSize)
for i = 1,numPatches do
   xlua.progress(i,numPatches)
   local r = torch.random(data_dim[2] - kSize + 1)
   local c = torch.random(data_dim[3] - kSize + 1)
   patches[i] = unlabeled_data[{((i-1) % 100000)+1,{},{r,r+kSize-1},{c,c+kSize-1}}]
   --patches[i] = provider.trainData.data[{math.fmod(i-1,trsize)+1,{},{r,r+kSize-1},{c,c+kSize-1}}]
   --normalization may not be needed here as done patch level. Need to experiment.
   --patches[i] = patches[i]:add(-patches[i]:mean())
   --patches[i] = patches[i]:div(math.sqrt(patches[i]:var()+10))
end

-- normalize patches
patches = patches:resize(numPatches, data_dim[1] * kSize * kSize)
mean = patches:mean(2)
std = (patches:var(2)+10):sqrt()
for j = 1, data_dim[1] * kSize * kSize do
   patches[{{}, {j}}]:add(-mean)
   patches[{{}, {j}}]:cdiv(std)
end
-- whiten
patches = unsup.zca_whiten(patches, nil, nil, nil, 1e-4)
patches = patches:resize(numPatches, data_dim[1], kSize, kSize)

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
   --print("Counts for iteration "..step)
   --print(c_counts)
   if itorch then
       itorch.image(c_kernels:index(1,varind[{{1,math.min(96,(#varind)[1])}}]))
   end
   resized_kernels = c_kernels
end

kernels, counts = unsup.kmeans(patches, ncentroids, 15, 10000, dispfilters, true)

print("==> select distinct features")
--resized_kernels = dispfilters(10,kernels,counts)

sfile = 'models/kmeans_'..ncentroids..'.t7'
print('==> saving centroids to disk: ' .. sfile)
obj = {
    resized_kernels = resized_kernels,
    kernels = kernels,
    counts = counts
}
torch.save(sfile, obj)
