cate2bin = function(namedList)
{
  thename = names(namedList)
  x = namedList[[1]]
  ux = sort(unique(x))
  y = integer(length(x))
  rst = as.data.frame(lapply(1:(length(ux) - 1L), function(i) {y[x == ux[i]] = 1L; y}))
  rownames(rst) = NULL
  colnames(rst) = paste0(thename, 0:(ncol(rst) - 1L)); rst
}


generateChunkInd = function(Nalldata, Nchunk)
{
  tmp = as.integer(round(seq(1, Nalldata + 1, len = Nchunk + 1)))
  rst = list()
  for(i in 1:(length(tmp) - 1L)) rst[[i]] = tmp[i]:(tmp[i + 1L] - 1L)
  rst
}


generateTrainDataSeq = function(Ndata, Npiece)
{
  tmp = as.integer(round(seq(1, Ndata, len = Npiece + 1L)))[-1]
  pieceList = list()
  for(i in 1:length(tmp)) pieceList[[i]] = 1L:tmp[i]
  pieceList
}











