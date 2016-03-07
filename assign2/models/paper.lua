require 'nn'

local model = nn.Sequential()
local poolsize = 2
local nstates = {96,64}
local filtsize = {13,5}
local group_size = 4

-- usually, we will get secondLayer from classifier.lua
if not secondLayer then
   secondLayer = {}
   for i=1, 24 do
      secondLayer[i] = nn.SpatialConvolution(group_size,nstates[2],filtsize[2],filtsize[2])
   end
end

model:add(nn.SpatialBatchNormalization(nstates[1]))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

model:add(nn.SpatialConvolution(nstates[1],nstates[1],1,1))
model:add(nn.Reshape(nstates[1]/group_size,group_size,10,10))

model:add(nn.Dropout(0.5))

multiconv = nn.Parallel(2,2)
for i = 1,(nstates[1]/group_size) do
    local seq_container = nn.Sequential()
    seq_container:add(secondLayer[i])
    if opt.trainsecond > 1 then
       seq_container:add(nn.SpatialBatchNormalization(nstates[2]))
    end
    seq_container:add(nn.ReLU())
    multiconv:add(seq_container)
end

model:add(multiconv)

model:add(nn.View(nstates[2]*(nstates[1]/group_size)*6*6))
model:add(nn.Linear(nstates[2]*(nstates[1]/group_size)*6*6, 10))
return model
