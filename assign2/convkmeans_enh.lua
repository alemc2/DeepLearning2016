require('unsup')
require('nn')

function unsup.convkmeans(provider,k, nch, kw, kh, nsamples, niter, batchsize, callback, verbose)
  local help = 'centroids,count = unsup.kmeans(Tensor(npoints,dim), k, kw, kh [, niter, batchsize, callback, verbose])'
  provider = provider or error('missing argument: ' .. help)
  k = k or error('missing argument: ' .. help)
  nch = nch or error('missing argument: ' .. help)
  kw = kw or error('missing argument: ' .. help)
  kh = kh or error('missing argument: ' .. help)
  nsamples = nsamples or error('missing argument: ' .. help)
  niter = niter or 1
  batchsize = batchsize or math.min(1000, nsamples)
  
  -- resize data
  local k_size = torch.LongStorage(4)
  k_size[1] = k
  k_size[2] = nch
  k_size[3] = kh
  k_size[4] = kw
  
  -- some shortcuts
  local sum = torch.sum
  local max = torch.max
  local pow = torch.pow
   
  -- dims
  local ndims = nch*kh*kw
   
  -- initialize means
  local centroids = torch.Tensor(k,ndims):normal()
  for i = 1,k do
     centroids[i]:div(centroids[i]:norm())
  end
  local totalcounts = torch.Tensor(k):zero()
     
  -- callback?
  if callback then callback(0,centroids:reshape(k_size),torch.ones(k)) end


  -- do niter iterations
  for i = 1,niter do
    -- progress
    if verbose then xlua.progress(i,niter) end

    -- init some variables
    local summation = torch.Tensor(k,ndims):zero()
    local counts = torch.Tensor(k):zero()

    -- process batch
    for i = 1,nsamples,batchsize do
       -- indices
       local lasti = math.min(i+batchsize-1,nsamples)
       local m = lasti - i + 1
       local batch,numpatches = provider:getData(i,m,kw,kh,nch)
       local pool = nn.TemporalMaxPooling(numpatches)

       -- k-means step, on minibatch
       local batch_patch = torch.Tensor(m,ndims)
       local batch_t = batch:t()
       local tmp = centroids * batch_t
       local val,labels = max(tmp,1)

       local pooled = pool:forward(val:t())
       local pooled_indices = pool.indices+1
       pooled_indices = pooled_indices[{{},1}] -- Convert to 1D tensor
       pooled_indices:add((torch.range(1,m)-1):mul(numpatches))
       
       --local win_batches = x[{ {i,lasti} }]
       --local numpatches = (kh+1)*(kw+1)
       --local batch_patch = torch.Tensor(m,ndims)
       --local S = x.new(m,k):zero()
       --for win_num = 1,m do
       --    local patches = torch.zeros(numpatches,ndims)
       --    --Extract patches, TODO: Check if normalized,whitened at this stage
       --    for r = 1,kh+1 do
       --        for c = 1,kw+1 do
       --            if x:dim()==4 then
       --                patches[(r-1)*(kw+1)+c] = win_batches[{ win_num, {}, {r, r+kh-1 }, {c, c+kw-1} }]:reshape(ndims)
       --            else
       --                patches[(r-1)*(kw+1)+c] = win_batches[{ win_num, {r, r+kh-1 }, {c, c+kw-1} }]
       --            end
       --            --Normalize the patches
       --            patches[(r-1)*(kw+1)+c]:add(-patches[(r-1)*(kw+1)+c]:mean())
       --            patches[(r-1)*(kw+1)+c]:div(math.sqrt(patches[(r-1)*(kw+1)+c]:var()+10))
       --        end
       --    end
       --    --whiten patches
       --    patches = unsup.zca_whiten(patches)
       --    --count exemplars per window
       --    local patch_t = patches:t()
       --    local tmp = centroids * patch_t
       --    local max_cent_val,max_cent_idx = max(tmp,1)
       --    local max_val,max_pnum =max(max_cent_val,2)
       --    max_pnum = max_pnum[1][1] --Weird tensor dimension problem
       --    batch_patch[win_num] = patches[max_pnum]
       --    S[win_num][max_cent_idx[1][max_pnum]] = max_val
       --    counts[max_cent_idx[1][max_pnum]] = counts[max_cent_idx[1][max_pnum]]+1
       --end

       -- count examplars per template
       local S = torch.Tensor(m,k):zero()
       for win_num = 1,m do
           S[win_num][labels[1][pooled_indices[win_num]]] = val[1][pooled_indices[win_num]]
           batch_patch[win_num] = batch[pooled_indices[win_num]]
       end
       summation:add( S:t() * batch_patch )
    end

    -- normalize
    centroids:add(summation); --Damped
    --centroids = summation; --No damping
    for i = 1,k do
       centroids[i]:div(centroids[i]:norm())
    end
    
    -- total counts
    totalcounts= counts:clone()

    -- callback?
    if callback then 
       local ret = callback(i,centroids:reshape(k_size),counts) 
       --if ret then break end
    end
  end

  -- done
  return centroids:reshape(k_size),totalcounts

end
