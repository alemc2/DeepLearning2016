require 'xlua'
require 'optim'
require 'unsup'
require 'ckmeans'
dofile './provider.lua'
c = require 'trepl.colorize'

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
pSize = 7
kSize = pSize * 2
trsize = 100000
numWindows = 300000
patches = torch.zeros(numWindows * (pSize + 1) * (pSize + 1), data_dim[1], pSize, pSize)
for i = 1,numWindows do
   xlua.progress(i,numWindows)
   local r = torch.random(data_dim[2] - kSize + 1)
   local c = torch.random(data_dim[3] - kSize + 1)
   window = unlabeled_data[{((i-1) % 100000)+1,{},{r,r+kSize-1},{c,c+kSize-1}}]
   for j = 1, pSize+1 do
      for k = 1,pSize+1 do
         patch = window[{ {}, {j,j+pSize-1}, {k,k+pSize-1} }]
         patch = patch:float()
         mean = patch:mean()
         std = math.sqrt(patch:var()+10)
         patch:add(-mean)
         patch:div(std)
         patches[1+(i-1)*(pSize+1)*(pSize+1)+(j-1)*(pSize+1)+(k-1)] = patch
      end
   end
end
unlabeled_data = nil
collectgarbage()

print(c.blue "==> whitening")
patches = patches:reshape(numWindows * (pSize + 1) * (pSize + 1), data_dim[1] * pSize * pSize)
count = numWindows * (pSize + 1) * (pSize + 1)
bsize = math.ceil(count / 15)
for i = 1,count, bsize do
   xlua.progress(i,count)
   lasti = math.min(i+bsize-1, count)
   patches[{{i,lasti}}] = unsup.zca_whiten(patches[{{i,lasti}}])
   collectgarbage()
end
patches = patches:reshape(numWindows * (pSize + 1) * (pSize + 1), data_dim[1], pSize, pSize)

collectgarbage()
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

kernels, counts = unsup.ckmeans(patches, ncentroids, (pSize + 1) * (pSize + 1), 15, 1000, dispfilters, true)
collectgarbage()
print("==> select distinct features")
--resized_kernels = dispfilters(10,kernels,counts)

sfile = 'models/ckmeans_'..ncentroids..'.t7'
print('==> saving centroids to disk: ' .. sfile)
obj = {
    resized_kernels = resized_kernels,
    kernels = kernels,
    counts = counts
}
torch.save(sfile, obj)
