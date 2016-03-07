require 'nn'

local model = nn.Sequential()
local poolsize = 4
local nstates = {96,64}
local filtsize = {13,5}
local group_size = 4

model:add(nn.SpatialBatchNormalization(nstates[1]))
model:add(nn.ReLU())
--model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

local mapping = nn.tables.random(nstates[1],nstates[1],24)
model:add(nn.SpatialConvolution(nstates[1],nstates[1],1,1))
model:add(nn.Reshape(nstates[1]/group_size,group_size,21,21))

multiconv = nn.Parallel(2,2)
for i = 1,(nstates[1]/group_size) do
    multiconv:add(nn.SpatialConvolution(group_size,nstates[2],filtsize[2],filtsize[2]))
end

model:add(multiconv)

model:add(nn.View(nstates[2]*(nstates[1]/group_size)*17*17))
--model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[2]*(nstates[1]/group_size)*17*17, 10))
return model
