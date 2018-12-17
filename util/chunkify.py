import numpy as np


# CHUNKIFY extract minibatch index
# return row indexes for chunks of a given size

def chunkify(chunksize, x):
    m, n = x.shape
    numchunks = np.ceil(m / chunksize)
    batchstart = 0
    batchend = chunksize

    chunk_start = []
    chunk_end = []
    # chunks = {
    #     'start': chunk_start,
    #     'end': chunk_end
    # }
    chunks = [None] * int(numchunks)
    # chunk_start = []
    # chunk_end = []

    for i in range(int(numchunks)):
        if batchend <= m:
            chunks[i] = {} # ['start'[i]] = batchstart
            chunks[i]['start'] = batchstart
            chunks[i]['end'] = batchend
        else:
            chunks[i] = {}
            chunks['start'][i] = batchstart
            chunks['end'][i] = m

        batchstart = batchend
        batchend = batchstart + chunksize

    return chunks


#
#