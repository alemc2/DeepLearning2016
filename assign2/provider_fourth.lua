require 'nn'
require 'image'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')

local Provider_Fourth = torch.class 'Provider_Fourth'

function Provider_Fourth:__init()
  local trsize = 100000

  self.fourthData = {
     data = torch.load('stl-10/fourthlevel.t7'),
     size = function() return trsize end
  }
  collectgarbage()
end

function Provider_Fourth:getData(start,bsize,kw,kh,nch)
    local fourth_data = self.fourthData
    local numpatches = (kh+1)*(kw+1)
    local ndims = nch*kh*kw
    local batches = torch.Tensor(bsize*numpatches,ndims)

    local trsize = 100000
    local height = 8
    local width = 8
    --print('getting data')
    for i = start, start+bsize-1 do
       local wr = torch.random(height - 2*kh + 1)
       local wc = torch.random(width - 2*kw + 1)
       local window = fourth_data.data[{math.fmod(i-1,trsize)+1,{},{wr,wr+2*kh-1},{wc,wc+2*kw-1}}]
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
    --print('whitening')
    batches = unsup.zca_whiten(batches)
    --print('returning')

    return batches,numpatches
end
