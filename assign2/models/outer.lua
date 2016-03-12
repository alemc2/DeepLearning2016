require 'nn'
require 'image'

do
   local RescaleModule,parent = torch.class('nn.RescaleModule', 'nn.Module')

   function RescaleModule:__init(scale_factor)
      parent.__init(self)
      self.scale_factor = scale_factor;
   end

   function RescaleModule:updateOutput(input)
      local bs = input:size(1)
      local nch = input:size(2)
      local imheight = input:size(3)
      local imwidth = input:size(4)
      self.output = torch.Tensor(bs,nch,imheight*self.scale_factor,imwidth*self.scale_factor)
      for i=1,bs do
         self.output[i] = image.scale(input[i],imwidth*self.scale_factor,imheight*self.scale_factor)
      end
      return self.output
  end
end

local outer = nn.Sequential()
local concatlayer = nn.Concat(3) -- 3 because of batch
local numscales = 3
local rescale_factors = {2/3,1/3}

for i in 1,numscales do
    local seq_container = nn.Sequential()
    seq_container:add(nn.RescaleModule(rescale_factors[i])) -- Rescaling
    seq_container:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
    seq_container:add(dofile('model/inner'..i..'.lua'):cuda())
    local cmul_layer = nn.CMul(10)
    cmul_layer.weight = torch.ones(10)
    seq_container:add(cmul_layer:cuda())
    concatlayer:add(seq_container)
end

outer:add(concatlayer)
outer:add(nn.Mean(3):cuda()) -- dim 3 for batch

return outer

