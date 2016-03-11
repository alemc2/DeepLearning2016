do -- data augmentation module
   local DataAugment,parent = torch.class('nn.DataAugment', 'nn.Module')

   function DataAugment:__init()
      parent.__init(self)
   end

   function DataAugment:updateOutput(input)
      --dummy module
      self.output:set(input)
      return self.output
   end
end
