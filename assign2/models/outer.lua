require 'nn'
require 'image'

do
   local RescaleModule,parent = torch.class('nn.RescaleModule', 'nn.Module')

   function RescaleModule:__init(scale_factor)
      parent.__init(self)
      self.scale_factor = scale_factor;
   end

   function RescaleModule:updateOutput(input)
      input = input:float()
      local bs = input:size(1)
      local nch = input:size(2)
      local imheight = input:size(3)
      local imwidth = input:size(4)
      self.output = torch.Tensor(bs,nch,imheight*self.scale_factor,imwidth*self.scale_factor)
      for i=1,bs do
         self.output[i] = image.scale(input[i],imwidth*self.scale_factor,imheight*self.scale_factor)
      end
      input = input:cuda()
      self.output = self.output:cuda()
      return self.output
   end
end

local outer = nn.Sequential()
local concatlayer = nn.Concat(3) -- 3 because of batch
local numscales = 1 -- TODO set to 3
local rescale_factors = {1,2/3,1/3}
local model_files = {'sample', 'sample64', 'sample32'}

for i = 1,numscales do
    local seq_container = nn.Sequential()
    seq_container:add(nn.RescaleModule(rescale_factors[i])) -- Rescaling
    seq_container:add(dofile('models/'..model_files[i]..'.lua'))
    local cmul_layer = nn.CMul(10)
    cmul_layer.weight = torch.ones(10)
    seq_container:add(cmul_layer)
    seq_container:add(nn.View(10,1))
    concatlayer:add(seq_container)
end

outer:add(concatlayer)
outer:add(nn.Mean(3)) -- dim 3 for batch
outer:add(nn.View(10))

return outer

