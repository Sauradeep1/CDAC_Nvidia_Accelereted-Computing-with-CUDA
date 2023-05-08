# Add your solution here
@cuda.jit
def cuda_histogram(x, xmin, xmax, histogram_out):
    '''Increment bin counts in histogram_out, given histogram range [xmin, xmax).'''
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins
    
    start = cuda.grid(1)
    stride = cuda.gridsize(1)   # 1 = one dimensional thread grid, returns a single value.

    for i in range(start, x.shape[0], stride):
    #for element in x:
        bin_number = np.int32((x[i] - xmin)/bin_width)
        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            # only increment if in range
            #histogram_out[bin_number] += 1
            cuda.atomic.add(histogram_out, bin_number, 1)
    #pass  # Replace this with your implementation