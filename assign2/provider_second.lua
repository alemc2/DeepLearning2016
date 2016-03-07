require 'nn'
require 'image'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')

local Provider_Second = torch.class 'Provider_Second'

function Provider_Second:__init(full)
  local groups = 24
  local trsize = 100000
  local channel = 4
  local height = 10
  local width = 10

  self.secondData = {
     data = torch.load('stl-10/secondlevel.t7'),
     size = function() return groups end
  }
  collectgarbage()
end

function Provider_Second:getData(group,start,bsize,kw,kh,nch)
    local second_data = self.secondData
    local numpatches = (kh+1)*(kw+1)
    local ndims = nch*kh*kw
    local batches = torch.Tensor(bsize*numpatches,ndims)
    
    local trsize = 100000
    local channel = 4
    local height = 10
    local width = 10
    
    for i = start, start+bsize-1 do
       local wr = torch.random(height - 2*kh + 1)
       local wc = torch.random(width - 2*kw + 1)
       local window = second_data.data[group][{math.fmod(i-1,trsize)+1,{},{wr,wr+2*kh-1},{wc,wc+2*kw-1}}]
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