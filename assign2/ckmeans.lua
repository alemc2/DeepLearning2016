require('unsup')
--
-- The k-means algorithm.
--
--   > x: is supposed to be an MxN matrix, where M is the nb of samples and each sample is N-dim
--   > k: is the number of kernels
--   > p: number of patches per window
--   > niter: the number of iterations
--   > batchmult: the number multiplied with p to get the batch size [large is good, to parallelize matrix multiplications]
--   > callback: optional callback, at each iteration end
--   > verbose: prints a progress bar...
--
--   < returns the k means (centroids) + the counts per centroid
--
function unsup.ckmeans(x, k, p, niter, batchmult, callback, verbose)
   -- args
   local help = 'centroids,count = unsup.ckmeans(Tensor(npoints,dim), k, p, [, niter, batchmult, callback, verbose])'
   x = x or error('missing argument: ' .. help)
   k = k or error('missing argument: ' .. help)
   p = p or error('missing argument: ' .. help)
   niter = niter or 1
   batchmult = batchmult or math.min(1000, (#x)[1])

   -- resize data
   local k_size = x:size()
   k_size[1] = k
   if x:dim() > 2 then
      x = x:reshape(x:size(1), x:nElement()/x:size(1))
   end

   -- some shortcuts
   local sum = torch.sum
   local max = torch.max
   local pow = torch.pow

   -- dims
   local nsamples = (#x)[1]
   local ndims = (#x)[2]

   -- initialize means
   local centroids = x.new(k,ndims):normal()
   for i = 1,k do
      centroids[i]:div(centroids[i]:norm())
   end
   local totalcounts = x.new(k):zero()
      
   -- callback?
   if callback then callback(0,centroids:reshape(k_size),totalcounts) end

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
      for i = 1,nsamples,(p*batchmult) do
         -- indices
         local lasti = i+(p*batchmult)-1

         -- k-means step, on minibatch
         local batch = x[{ {i,lasti},{} }]
         local batch_t = batch:t()
         local tmp = centroids * batch_t
         for n = 1,(#batch)[1] do
            tmp[{ {},n }]:add(-1,c2)
         end
         local val,labels = max(tmp,1)

         -- select one patch per window
         val = val:reshape(batchmult, p):t()
         val, index = max(val,1)
         labels = labels:reshape(batchmult, p):t()
         labels = labels:gather(1, index)

         index = index:reshape(batchmult,1):expand(batchmult,ndims):reshape(batchmult,1,ndims)
         batch = batch:reshape(batchmult, p, ndims)
         batch = batch:gather(2, index)
         batch = batch:reshape(batchmult, ndims)

         -- count examplars per template
         local S = x.new(batchmult,k):zero()
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
         local ret = callback(i,centroids:reshape(k_size),totalcounts) 
         if ret then break end
      end
      collectgarbage()
   end

   -- done
   return centroids:reshape(k_size),totalcounts
end
