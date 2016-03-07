require 'nn'

local model = nn.Sequential()

model:add(nn.SpatialBatchNormalization(96))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.SpatialConvolution(96, 192, 3, 3))
model:add(nn.SpatialBatchNormalization(192))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.View(192*4*4))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(192*4*4, 10))

return model
