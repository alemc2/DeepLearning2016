require 'nn'
require 'image'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')

-- parse STL-10 data from table into Tensor
function parseDataUnlabel(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local idx = 1
   for i = 1, #d do
      local this_d = d[i]
      for j = 1, #this_d do
    t[idx]:copy(this_d[j])
    idx = idx + 1
      end
   end
   assert(idx == numSamples+1)
   return t
end


local Provider_Unlabel = torch.class 'Provider_Unlabel'

function Provider_Unlabel:__init(full)
  local trsize = 100000
  local channel = 3
  local height = 96
  local width = 96

  -- download dataset
  if not paths.dirp('stl-10') then
     os.execute('mkdir stl-10')
     local www = {
         extra = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/extra.t7b',
     }

     os.execute('wget ' .. www.extra .. '; '.. 'mv extra.t7b stl-10/extra.t7b')
  end

  local raw_unlabel = torch.load('stl-10/extra.t7b')

  -- load and parse dataset
  self.unlabeledData = {
     data = torch.Tensor(),
     size = function() return trsize end
  }
  self.unlabeledData.data = parseDataUnlabel(raw_unlabel.data, trsize, channel, height, width)

  collectgarbage()
end

function Provider_Unlabel:getData(start,bsize,kw,kh,nch)
    local unlabeled_data = self.unlabeledData
    local numpatches = (kh+1)*(kw+1)
    local ndims = nch*kh*kw
    local batches = torch.Tensor(bsize*numpatches,ndims)
    
    local trsize = 100000
    local channel = 3
    local height = 96
    local width = 96
    
    for i = start, start+bsize-1 do
       local wr = torch.random(height - 2*kh + 1)
       local wc = torch.random(width - 2*kw + 1)
       local window = unlabeled_data.data[{math.fmod(i-1,trsize)+1,{},{wr,wr+2*kh-1},{wc,wc+2*kw-1}}]
       for r = 1,kh+1 do
           for c = 1,kw+1 do
               batches[(i-start)*numpatches + (r-1)*(kw+1) + c] = window[{ {}, {r, r+kh-1 }, {c, c+kw-1} }]:reshape(ndims)

               --Normalize the patches
               batches[(r-1)*(kw+1)+c]:add(-batches[(r-1)*(kw+1)+c]:mean())
               batches[(r-1)*(kw+1)+c]:div(math.sqrt(batches[(r-1)*(kw+1)+c]:var()+10))
           end
       end
    end

    --whiten patches
    batches = unsup.zca_whiten(batches)

    return batches,numpatches
end
