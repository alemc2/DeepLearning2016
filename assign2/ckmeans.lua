require('unsup')
--
-- The k-means algorithm.
--
--   > x: is supposed to be an MxN matrix, where M is the nb of samples and each sample is N-dim
--   > k: is the number of kernels
--   > niter: the number of iterations
--   > batchsize: the batch size [large is good, to parallelize matrix multiplications]
--   > callback: optional callback, at each iteration end
--   > verbose: prints a progress bar...
--
--   < returns the k means (centroids) + the counts per centroid
--
function unsup.ckmeans(x, k, kh, kw, niter, batchsize, callback, verbose)
   -- args
   local help = 'centroids,count = unsup.kmeans(Tensor(npoints,dim), k, kh, kw [, niter, batchsize, callback, verbose])'
   x = x or error('missing argument: ' .. help)
   k = k or error('missing argument: ' .. help)
   kh = kh or error('missing argument: ' .. help)
   kw = kw or error('missing argument: ' .. help)
   niter = niter or 1
   batchsize = batchsize or math.min(1000, (#x)[1])

   -- resize data
   if x:dim() == 3 then
      x = x:reshape(x:size(1), 1, x:size(2), x:size(3))
   end
   local k_size = x:size()
   k_size[1] = k
   k_size[3] = kh
   k_size[4] = kw

   -- some shortcuts
   local sum = torch.sum
   local max = torch.max
   local pow = torch.pow

   -- dims
   local nsamples = (#x)[1]
   local nchannels = (#x)[2]
   local ndims = nchannels*kh*kw

   -- initialize means
   local centroids = x.new(k,ndims):normal()
   for i = 1,k do
      centroids[i]:div(centroids[i]:norm())
   end
   local totalcounts = x.new(k):zero()

   -- callback?
   if callback then callback(0,centroids:reshape(k_size),torch.ones(k)) end

   -- do niter iterations
   for i = 1,niter do
      -- progress
      if verbose then xlua.progress(i,niter) end

      -- sums of squares
      local c2 = sum(pow(centroids,2),2)*0.5

      -- init some variables
      local summation = x.new(k,ndims):zero()
      local counts = x.new(k):zero()

      -- process batch
      for i = 1,nsamples,batchsize do
         -- indices
         local lasti = math.min(i+batchsize-1,nsamples)
         local m = lasti - i + 1

         -- k-means step, on minibatch
         local batch_x = x[{ {i,lasti} }]
         local batch = x.new(m, ndims)
         local val = x.new(m,1)
         local labels = x.new(m,1)
         for j = 1,m do
            local patches = x.new((kw+1)*(kh+1), ndims)
            for h = 1,kh+1 do
               for w = 1,kw+1 do
                  patches[1+((h-1)*(kw+1))+(w-1)] = batch_x[{j, {}, {h,h+kh-1}, {w,w+kw-1}}]:reshape(ndims)
               end
            end
            -- normalize patches
            mean = patches:mean(2)
            std = (patches:var(2)+10):sqrt()
            for j = 1, ndims do
               patches[{{}, {j}}]:add(-mean)
               patches[{{}, {j}}]:cdiv(std)
            end
            -- whiten
            patches = unsup.zca_whiten(patches, nil, nil, nil, 1e-4)

            local patches_t = patches:t()
            local tmp = centroids * patches_t
            for n = 1,(#patches)[1] do
               tmp[{ {},n }]:add(-1,c2)
            end
            local val1,index1 = max(tmp,1)
            local val2,index2 = max(val1,2)
            val[j] = val2[1][1]
            labels[j] = index1[1][index2[1][1]]
            batch[j] = patches[index2[1][1]]
         end

         val = val:t()
         labels = labels:t()

         -- count examplars per template
         local S = x.new(m,k):zero()
         for i = 1,(#labels)[2] do
            S[i][labels[1][i]] = 1
         end
         summation:add( S:t() * batch )
         counts:add( sum(S,1) )
      end

      -- normalize
      for i = 1,k do
         if counts[i] ~= 0 then
            centroids[i] = summation[i]:div(counts[i])
         end
      end

      -- total counts
      totalcounts:add(counts)

      -- callback?
      if callback then 
         local ret = callback(i,centroids:reshape(k_size),counts) 
         --if ret then break end
      end
      collectgarbage()
   end

   -- done
   return centroids:reshape(k_size),totalcounts
end
