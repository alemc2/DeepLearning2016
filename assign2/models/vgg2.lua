require 'nn'

local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane, size, stride)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, size, size, stride,stride, math.floor(size/2),math.floor(size/2)))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end

local function ConvBNReLUPretrain(nOutputPlane, layer)
  vgg:add(layer)
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling

ConvBNReLUPretrain(64,firstLayer):add(nn.Dropout(0.3))
ConvBNReLU(64,64,7,1)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(64,128,5,1):add(nn.Dropout(0.4))
ConvBNReLU(128,128,5,1)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(128,256,5,1):add(nn.Dropout(0.4))
ConvBNReLU(256,256,5,1):add(nn.Dropout(0.4))
ConvBNReLU(256,256,5,1)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(256,512,3,1):add(nn.Dropout(0.4))
ConvBNReLU(512,512,3,1):add(nn.Dropout(0.4))
ConvBNReLU(512,512,3,1)
vgg:add(MaxPooling(2,2,2,2):ceil())
vgg:add(nn.View(512*3*3))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512*3*3,512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,10))
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
