require 'nn'

local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling

ConvBNReLU(3,64)
vgg:add(MaxPooling(2,2,2,2):ceil()) -- 64x16x16

ConvBNReLU(64,128)
vgg:add(MaxPooling(2,2,2,2):ceil()) -- 128x8x8

ConvBNReLU(128,256):add(nn.Dropout(0.1))
ConvBNReLU(256,256)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(256,256):add(nn.Dropout(0.2))
ConvBNReLU(256,256):add(nn.Dropout(0.2))
ConvBNReLU(256,256)
vgg:add(MaxPooling(2,2,2,2):ceil())

vgg:add(nn.View(256*2*2))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*2*2,256))
classifier:add(nn.BatchNormalization(256))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.4))
classifier:add(nn.Linear(256,10))
classifier:add(nn.SoftMax())
vgg:add(classifier)

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return vgg
