require 'xlua'
require 'optim'
dofile './provider.lua'
dofile './convkmeans.lua'
local c = require 'trepl.colorize'

print(c.blue '==>' ..' loading data')
--provider = torch.load 'provider.t7'
--provider.trainData.data = provider.trainData.data:float()
--provider.valData.data = provider.valData.data:float()
raw_unlabel = torch.load('stl-10/extra.t7b')
unlabeled_data,_ = parseDataLabel(raw_unlabel.data,100000,3,96,96)
raw_unlabel = nil
collectgarbage()
for i = 1,unlabeled_data:size(1) do
    --Convert this to YUV as we do the same for labeled data but need to investigate if normalization needed
    unlabeled_data[i] = image.rgb2yuv(unlabeled_data[i])
end

print(c.blue "==> extract windows")
data_dim = {3,96,96}
kSize = 10
trsize = 4000
numPatches = 100000
patches = torch.zeros(numPatches, data_dim[1], kSize, kSize)
for i = 1,numPatches do
   xlua.progress(i,numPatches)
   local r = torch.random(data_dim[2] - kSize + 1)
   local c = torch.random(data_dim[3] - kSize + 1)
   patches[i] = unlabeled_data[{math.fmod(i-1,trsize)+1,{},{r,r+kSize-1},{c,c+kSize-1}}]
   --normalization may not be needed here as done patch level. Need to experiment.
   --patches[i] = patches[i]:add(-patches[i]:mean())
   --patches[i] = patches[i]:div(math.sqrt(patches[i]:var()+10))
end

print(c.blue "==> find clusters")
ncentroids = 1600
kernels, counts = unsup.convkmeans(patches, ncentroids, 5, 5, 10, 1000, nil, true)

print("==> select distinct features")
local j = 0
local variance = torch.Tensor(counts:size())
for i = 1,ncentroids do
   if counts[i] > 0 then
      j = j + 1
      kernels[{j,{}}] = kernels[{i,{}}]
      counts[j] = counts[i]
      variance[j] = kernels[{j,{}}]:var()
   end
end
kernels = kernels[{{1,j},{}}]
counts  = counts[{{1,j}}]
variance  = variance[{{1,j}}]

_,varind = torch.sort(variance)
if itorch then
    itorch.image(kernels:index(1,varind[{{1,50}}]))
end

sfile = 'models/ckmeans_'..ncentroids..'.t7'
print('==> saving centroids to disk: ' .. sfile)
torch.save(file, kernels)
