

source("r/rfuns.r")
tmp = data.table::setDF(data.table::fread("data/optdigits.tra", header = F))
tmp2 = data.table::setDF(data.table::fread("data/optdigits.tes", header = F))
dat = rbind(tmp, tmp2)
dat$y = dat$V65
dat$V65 = NULL
tmp = range(unlist(dat[-ncol(dat)]))
for(i in 1:(length(dat) -1L)) dat[[i]] = (dat[[i]] - tmp[1]) / tmp[2]
set.seed(123)
dat = dat[sample(nrow(dat)), ]


cvNfold = 10L
valiIndList = generateChunkInd(nrow(dat), cvNfold)
pieceList = generateTrainDataSeq(nrow(dat), Npiece = 20)
dataSizes = unlist(lapply(pieceList, function(x) length(x)))




# Support vector machine.
if(T)
{
  
  
  X = as.matrix(dat[-ncol(dat)])
  # X = dat[-ncol(dat)]
  # X = as.matrix(as.data.frame(lapply(X, function(x) (x - mean(x)) / sd(x))))
  # X[is.nan(X)] = 0
  
  
  # Gaussian kernel.
  if(T)
  {
    
    
    rbfXs = lapply(c(1e-3, 1e-4, 1e-5, 1e-6), function(s) kernlab::kernelMatrix(kernlab::rbfdot(sigma = s), X))
    lambda = unique(c(seq(0.01, 0.1, by = 0.01), seq(0.1, 1, by = 0.1), seq(1, 10, by = 1), seq(10, 100, by = 10)))
    
    
    maxCore = 8
    tmp = as.integer(round(seq(1, length(lambda) + 1, len = maxCore + 1)))
    tmplist = list()
    for(i in 1:(length(tmp)-1L)) tmplist[[i]] = lambda[tmp[i]:(tmp[i + 1L] - 1L)]
    computelist = lapply(tmplist, function(x)
    {
      list(lambda = x, rbfXs = rbfXs, y = dat$y)
    })
    
    
    cl = snow::makeCluster(maxCore, type = "SOCK")
    rbfKernelRst = unlist(snow::clusterApply(cl, computelist, function(X)
    {
      y = as.factor(X$y)
      lapply(X$lambda, function(lambda)
      {
        lapply(X$rbfXs, function(x)
        {
          set.seed(42)
          kernlab::ksvm(x = x, y = y, scaled = F, type = "C-svc", kernel = "matrix", C = lambda, cross = 10, cache = 2048 * 4)
        })
      })
    }), recursive = F)
    snow::stopCluster(cl) 
    save(rbfKernelRst, file = "data/rbfKernelRstDigit.Rdata")
    
    
    # optimal cv error = 0.0183274
    pdf("figure/rbfKernelRstCvErrInitialDigit.pdf", width = 9, height = 9 * 0.618)
    par(mar = c(4, 5, 0, 0), family = "serif")
    tmp = lapply(1:length(rbfKernelRst[[1]]), function(i) unlist(lapply(rbfKernelRst, function(x) x[[i]]@cross)))
    ylim = range(unlist(tmp)); xlim = c(1, length(tmp[[1]]))
    plot(0, xlim = xlim, ylim = ylim, col = "white", bty = "L", xlab = expression(lambda), ylab = "Cross validation error", xaxt = "n", cex.lab = 2, cex.axis = 1.5)
    axis(side = 1, at = 1:length(tmp[[1]]), labels = lambda, cex.axis = 1.5)
    linecols = c("black", "red", "blue", "olivedrab3")
    legend("left", legend = c(expression(sigma == 0.001), expression(sigma == 0.0001), expression(sigma == 0.00001), expression(sigma == 0.000001)), col = linecols, pch = c(15, 15, 15, 15), cex = 1.5, bty = "n")
    for(i in 1:length(tmp))
    {
      lines(tmp[[i]], col = linecols[i], lwd = 2)
    }
    dev.off()
    
    
  }
  
  
  
  
  # Laplacian kernel.
  if(T)
  {
    
    
    lapXs = lapply(c(1e-1, 1e-2, 1e-3, 1e-4), function(s) kernlab::kernelMatrix(kernlab::laplacedot(sigma = s), X))
    lambda = unique(c(seq(0.01, 0.1, by = 0.01), seq(0.1, 1, by = 0.1), seq(1, 10, by = 1), seq(10, 100, by = 10)))
    
    
    maxCore = 8
    tmp = as.integer(round(seq(1, length(lambda) + 1, len = maxCore + 1)))
    tmplist = list()
    for(i in 1:(length(tmp)-1L)) tmplist[[i]] = lambda[tmp[i]:(tmp[i + 1L] - 1L)]
    computelist = lapply(tmplist, function(x)
    {
      list(lambda = x, lapXs = lapXs, y = dat$y)
    })
    
    
    cl = snow::makeCluster(maxCore, type = "SOCK")
    lapKernelRst = unlist(snow::clusterApply(cl, computelist, function(X)
    {
      y = as.factor(X$y)
      lapply(X$lambda, function(lambda)
      {
        lapply(X$lapXs, function(x)
        {
          set.seed(42)
          kernlab::ksvm(x = x, y = y, scaled = F, type = "C-svc", kernel = "matrix", C = lambda, cross = 10, cache = 2048 * 4)
        })
      })
    }), recursive = F)
    snow::stopCluster(cl) 
    save(lapKernelRst, file = "data/lapKernelRstDigit.Rdata")
    
    
    pdf("figure/lapKernelRstCvErrInitialDigit.pdf", width = 9, height = 9 * 0.618)
    par(mar = c(4, 5, 0, 0), family = "serif")
    tmp = lapply(1:length(lapKernelRst[[1]]), function(i) unlist(lapply(lapKernelRst, function(x) x[[i]]@cross)))
    ylim = range(unlist(tmp)); xlim = c(1, length(tmp[[1]]))
    plot(0, xlim = xlim, ylim = ylim, col = "white", bty = "L", xlab = expression(lambda), ylab = "Cross validation error", xaxt = "n", cex.lab = 2, cex.axis = 1.5)
    axis(side = 1, at = 1:length(tmp[[1]]), labels = lambda, cex.axis = 1.5)
    linecols = c("black", "red", "blue", "olivedrab3")
    legend("topright", legend = c(expression(sigma == 0.1), expression(sigma == 0.01), expression(sigma == 0.001), expression(sigma == 0.0001)), col = linecols, pch = c(15, 15, 15, 15), cex = 1.5, bty = "n")
    for(i in 1:length(tmp))
    {
      lines(tmp[[i]], col = linecols[i], lwd = 2)
    }
    legend("right", legend = paste0("min(err) = ", round(min(unlist(tmp)), 5)), cex = 1.5, bty = "n")
    dev.off()
  }
  
  
  

  # Plot learning curve.
  if(T)
  {
    
    
    trainValidErr = as.data.frame(t(as.data.frame(lapply(pieceList, function(x)
    {
      cat(".")
      alldat = dat[x, ]
      lapX = kernlab::kernelMatrix(kernlab::laplacedot(sigma = 0.1), as.matrix(alldat[-ncol(alldat)]))
      set.seed(42)
      mdl = kernlab::ksvm(x = lapX, y = as.factor(alldat$y), scaled = F, type = "C-svc", kernel = "matrix", C = 100, cross = 10, cache = 2048 * 4)
      c(mdl@error, mdl@cross)
    }))))
    rownames(trainValidErr) = NULL
    colnames(trainValidErr) = c("trainErr", "validErr")
    
    
    learningCurve = data.frame(dataSize = unlist(lapply(pieceList, function(x) length(x))), trainValidErr)
    tmp = learningCurve
    pdf("figure/lapKernelSVMlearningCurveDigit.pdf", width = 9, height = 9 * 0.618)
    par(mar = c(4, 5, 0, 0), family = "serif")
    plot(x = tmp$dataSize, y = tmp$trainErr, ylim = range(unlist(tmp[-1])), type = "l", col = "red", xlab = "N(observation)", ylab = "Error", bty = "L", cex.lab = 2, cex.axis = 1.5, lwd = 2)
    legend("topright", legend = c("Cross validation error", "Training error"), col = c("darkblue", "red"), cex = 2, lwd = c(2, 2), bty = "n")
    lines(x = tmp$dataSize, y = tmp$validErr, col = "darkblue", lwd = 2)
    dev.off()
    save(learningCurve, file = "data/lapKernelSVMlearingCurveDigit.Rdata")
  }
  
  
}




# Single decision tree.
if(T)
{
  
  
  # See learning curve
  set.seed(1)
  fml = paste0(colnames(dat)[ncol(dat)], "~", paste0(colnames(dat)[-ncol(dat)], collapse = "+"))
  fml = eval(parse(text = fml))
  dtreeLearned = lapply(pieceList, function(x)
  {
    cat(".")
    alldat = dat[x, ]
    alldat$y = as.factor(alldat$y)
    tmptree = rpart::rpart(fml, data = alldat, method = "class", control = rpart::rpart.control(minsplit = 0, cp = 0, xval = 10))
    cp = tmptree$cptable[which.min(tmptree$cptable[, "xerror"]), "CP"]
    # finaltree = rpart::prune.rpart(tmptree, cp = cp)
    finaltree = tmptree
    valilist = generateChunkInd(nrow(alldat), Nchunk = 10)
    trainValidErr = as.data.frame(t(as.data.frame(lapply(valilist, function(u)
    {
      train = alldat[-u, ]
      valid = alldat[u, ]
      tmptree = rpart::rpart(fml, data = train, method = "class", control = rpart::rpart.control(minsplit = 0, cp = cp, xval = 0))
      trainPred = predict(tmptree, newdata = train, "class")
      trainErr = sum(trainPred != train[[ncol(train)]]) / nrow(train)
      validErr = sum(predict(tmptree, newdata = valid, "class") != valid[[ncol(valid)]]) / nrow(valid)
      c(trainErr, validErr)
    }))))
    rownames(trainValidErr) = NULL
    colnames(trainValidErr) = c("trainErr", "validErr")
    list(model = finaltree, cvErr = trainValidErr)
  })
  
  
  # opt tree size = 324, opt cp = 0.004358162
  pdf("figure/dtreeDigitCp.pdf", width = 9, height = 9 * 0.618)
  par(mar = c(4.5, 5, 5, 0), family = "serif", cex.lab = 2, cex.axis = 1.5)
  rpart::plotcp(dtreeLearned[[length(dtreeLearned)]]$model)
  legend("topright", bty = "n", cex = 1.5, legend = c("Optimal tree size = 324", "Optimal cp = 0.00436"))
  dev.off()
  
  
  tmp = as.data.frame(t(as.data.frame(lapply(dtreeLearned, function(x) colMeans(x$cvErr)))))
  rownames(tmp) = NULL; colnames(tmp) = c("trainErr", "validErr")
  
  
  pdf("figure/dtreeLearningCurveDigit.pdf", width = 9, height = 9 * 0.618)
  par(mar = c(4, 5, 0, 0), family = "serif")
  plot(x = dataSizes, y = tmp$trainErr, ylim = range(unlist(tmp)), type = "l", col = "red", xlab = "N(observation)", ylab = "Error (misclassification rate)", bty = "L", cex.lab = 2, cex.axis = 1.5, lwd = 2)
  legend("topright", legend = c("Cross validation error", "Training error", "Full data CV error = 0.088"), col = c("darkblue", "red", "white"), cex = 2, lwd = c(2, 2, 1), bty = "n")
  lines(x = dataSizes, y = tmp$validErr, col = "darkblue", lwd = 2)
  dev.off()
  save(dtreeLearned, file = "data/dtreeLearingCurveDigit.Rdata")
  
  
}




# xgboost
if(T)
{
  # Find optimal number of trees and learning rate via cross-validation.
  if(T)
  {
    
    
    eta = seq(0.01, 1, len = 100)
    rst = list()
    tmpdat = as.matrix(dat[-ncol(dat)])
    for(i in 1:length(eta))
    {
      cat(i, "")
      set.seed(42)
      rst[[i]] = xgboost::xgb.cv(params = list(objective = "multi:softprob", eta = eta[i], max_depth = 6, nthread = 13, num_class = length(unique(dat$y))), data = tmpdat, nrounds = 100, metrics = "merror", label = dat$y, nfold = 10, verbose = F)
    }
    rm(tmpdat); gc()
    save(rst, file = "data/xgboostRstDigit.Rdata")
    
    
    tmp = as.data.frame(lapply(rst, function(x) x$evaluation_log$test_merror_mean))
    colnames(tmp) = paste0("lr", eta); rownames(tmp) = NULL
    optLrate = (as.integer(which.min(unlist(tmp)) / 100) + 1L) / 100
    optNtree = as.integer(which.min(unlist(tmp))) %% 100L
    tmp = t(tmp)
    minErr = min(tmp)
    maxErr = max(tmp)
    tmp = as.matrix(tmp[nrow(tmp):1, ])
    dimnames(tmp) = NULL
    tmp[order(tmp)] = qnorm(seq(0.005, 0.995, len = length(tmp)))
    pdf("figure/xgboostLrNtreeDigit.pdf", width = 10, height = 8)
    par(mar = c(4, 5, 0, 0), family = "serif")
    image(t(tmp)[, nrow(tmp):1], col = unique(colorRampPalette(c("blue", "white", "red"))(20000)), xlab = "N(tree)", ylab = "Learning rate", cex.lab = 2, xaxt = "n", yaxt = "n", xlim = c(0, 1.2), bty = "n")
    axis(side = 1, at = seq(0, 1, len = 100), labels = 1:100, cex.axis = 1.5)
    axis(side = 2, at = seq(0, 1, len = length(eta)), labels = eta, cex.axis = 1.5)
    lines(x = rep(1.05, 100), y = seq(0.3, 0.7, len = 100), pch = 15, col = colorRampPalette(c("blue", "white", "red"))(100), type = "p")
    text(x = 1.11, y = 0.3, labels = paste0("Low\n", round(minErr, 5)), cex = 1.5)
    text(x = 1.11, y = 0.75, labels = paste0("High error\n", round(maxErr, 5)), cex = 1.5)
    lines(x = optNtree / 100 - 0.01, y = optLrate - 0.01, cex = 3, type = "p", col = "black", lwd = 5)
    text(x = optNtree / 100 - 0.01 + 0.05, y = optLrate - 0.01 + 0.05, labels = paste0("(", optNtree, ", ", optLrate, ")"), cex = 2)
    dev.off()
    
    
  }
  
  
  
  
  # Plot learning curve in terms of training data.
  if(T)
  {
    
    
    trainValidErr = as.data.frame(t(as.data.frame(lapply(pieceList, function(x)
    {
      cat(".")
      alldat = dat[x, ]
      tmp = xgboost::xgb.cv(params = list(objective = "multi:softprob", eta = optLrate, max_depth = 6, nthread = 13, num_class = length(unique(dat$y))), data = as.matrix(alldat[-ncol(alldat)]), nrounds = optNtree, metrics = "merror", label = alldat[[ncol(alldat)]], nfold = 10, verbose = F)
      n = length(tmp$evaluation_log$test_merror_mean)
      c(tmp$evaluation_log$train_merror_mean[n], tmp$evaluation_log$test_merror_mean[n])
    }))))
    rownames(trainValidErr) = NULL
    colnames(trainValidErr) = c("trainErr", "validErr")
    
    
    learningCurve = data.frame(dataSize = unlist(lapply(pieceList, function(x) length(x))), trainValidErr)
    tmp = learningCurve
    pdf("figure/xgboostLearningCurveDigit.pdf", width = 9, height = 9 * 0.618)
    par(mar = c(4, 5, 0, 0), family = "serif")
    plot(x = tmp$dataSize, y = tmp$trainErr, ylim = range(unlist(tmp[-1])), type = "l", col = "red", xlab = "N(observation)", ylab = "Error", bty = "L", cex.lab = 2, cex.axis = 1.5, lwd = 2)
    legend("topright", legend = c("Cross validation error", "Training error"), col = c("darkblue", "red"), cex = 2, lwd = c(2, 2), bty = "n")
    lines(x = tmp$dataSize, y = tmp$validErr, col = "darkblue", lwd = 2)
    dev.off()
    save(learningCurve, file = "data/xgboostLearingCurveDigit.Rdata")
    
    
  }
}








# K-nearest neighbor
if(T)
{
  
  
  dmat = as.matrix(dist(as.matrix(dat[-ncol(dat)])))
  dimnames(dmat) = NULL
  dmats = lapply(pieceList, function(x) as.data.frame(dmat[x, x]))
  
  
  knnCVs = list()
  for(k in 1:length(dmats))
  {
    
    
    cat(k, "")
    dmat = dmats[[k]]
    K = 50L
    
    
    rst = list()
    tmpValiIndList = generateChunkInd(nrow(dmat), 10)
    for(i in 1:length(tmpValiIndList))
    {
      I = tmpValiIndList[[i]]
      rst[[i]] = lapply(dmat[I], function(x) # a list, the i_th element
      {
        x = x[-I]
        y = dat$y[-I]
        nns = order(x)[2:(K + 1L)]
        nnYs = y[nns]
        sapply(1:K, function(k) 
        {
          tmp = table(nnYs[1:k])
          as.integer(names(tmp)[which.max(tmp)])
        })
        # nnYs
        # tmp = table(nnYs)
        # as.integer(names(tmp)[which.max(tmp)])
        # tmp = as.integer(sign(cumsum(nnYs)))
        # tmp[tmp == 0L] = 1L
        # tmp
      })
    }
    
    
    cvErr = list()
    for(i in 1:length(rst))
    {
      tmp = as.data.frame(t(as.data.frame(rst[[i]])))
      cvErr[[i]] = unlist(lapply(tmp, function(x)
      {
        sum(x != dat$y[tmpValiIndList[[i]]]) / length(x)
      }))
      names(cvErr[[i]]) = NULL
    }
    cvErr = as.data.frame(t(as.data.frame(cvErr)))
    rownames(cvErr) = NULL
    colnames(cvErr) = paste0("NN", 1:ncol(cvErr))
    knnCVs[[k]] = cvErr
  }
  
  
  cvErr = knnCVs[[length(knnCVs)]]
  pdf("figure/cvErrKNNdigit.pdf", width = 9, height = 9 * 0.618)
  par(mar = c(4, 5, 0, 0), family = "serif")
  plot(0, col = "white", xlim = c(1, length(cvErr)), ylim = range(unlist(cvErr)), bty = "L", cex.lab = 2, cex.axis = 1.5, ylab = "Cross validation error", xlab = "K", xaxt = "n")
  axis(side = 1, at = 1:length(cvErr), labels = 1:length(cvErr), cex.axis = 1.5)
  for(i in 1:length(cvErr))
  {
    m = mean(cvErr[[i]]); s = sd(cvErr[[i]])
    lines(x = i, y = mean(cvErr[[i]]), type = "p", col = "darkblue")
    lines(x = c(i, i), y = c(m - s, m + s))
  }
  legend("topright", legend = c("min(err) = 0.016", "opt(K) = 7"), bty = "n", cex = 2)
  dev.off()
  
  
  tmp = unlist(lapply(knnCVs, function(x) min(colMeans(x))))
  pdf("figure/KNNlearningCurveDigit.pdf", width = 9, height = 9 * 0.618)
  par(mar = c(4, 5, 0, 0), family = "serif")
  plot(x = dataSizes, y = tmp, bty = "L", cex.lab = 2, cex.axis = 1.5, ylab = "Cross validation error", xlab = "Data size", xaxt = "s", type = "l", lwd = 2)
  dev.off()
  
  
}




# Neural nets.
if(T)
{
  
  
  # keras::layer_flatten(object, data_format = NULL, input_shape = NULL, dtype = NULL, name = NULL, trainable = NULL, weights = NULL)
  # keras::layer_dense(object, units, activation = NULL, use_bias = TRUE, kernel_initializer = "glorot_uniform", bias_initializer = "zeros", kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL, kernel_constraint = NULL, bias_constraint = NULL, input_shape = NULL, batch_input_shape = NULL, batch_size = NULL, dtype = NULL, name = NULL, trainable = NULL, weights = NULL)
  
  
  # Preliminary tests
  if(F)
  {
    tmpX = as.matrix(dat[, -ncol(dat)])
    tmpY = dat$y
    
    
    model = keras::keras_model_sequential()
    keras::layer_flatten(model, input_shape = ncol(tmpX))
    
    
    keras::layer_dense(model, units = 64, activation = NULL) # "relu"
    keras::layer_activation_leaky_relu(model, alpha = 0.01)
    keras::layer_dropout(model, rate = 0)
    
    
    keras::layer_dense(model, units = 64, activation = NULL) # "relu"
    keras::layer_activation_leaky_relu(model, alpha = 0.01)
    keras::layer_dropout(model, rate = 0)
    
    
    keras::layer_dense(model, 10, activation = "softmax")
    
    
    opt = keras::optimizer_adam(lr = 0.0005)
    keras::compile(model, loss = "sparse_categorical_crossentropy", optimizer = opt, metrics = "accuracy")
    system.time({fm = keras::fit(model, x = tmpX, y = tmpY, batch_size = 32, epochs = 200, validation_split = 0.3, verbose = 2, shuffle = T)}) # https://keras.rstudio.com/reference/fit.html
    
    
  }
  
  
  # Find best learning rate, and plot.
  if(T)
  {
    
    
    tmpX = as.matrix(dat[, -ncol(dat)])
    tmpY = dat$y
    
    
    lrlist = c(0.1, 0.01, 0.001, 0.0001)
    modelViaDiffLr = lapply(lrlist, function(r)
    {
      model = keras::keras_model_sequential()
      keras::layer_flatten(model, input_shape = ncol(tmpX))
      
      
      keras::layer_dense(model, units = 32, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0.5)
      
      
      keras::layer_dense(model, units = 32, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0.5)
      
      
      keras::layer_dense(model, 10, activation = "softmax")
      
      
      opt = keras::optimizer_adam(lr = r)
      keras::compile(model, loss = "sparse_categorical_crossentropy", optimizer = opt, metrics = "accuracy")
      system.time({fm = keras::fit(model, x = tmpX, y = tmpY, batch_size = 32, epochs = 100, validation_split = 0.3, verbose = 2, shuffle = T)}) # https://keras.rstudio.com/reference/fit.html
      fm
    })
    
    
    # Train-valid split 0.3
    pdf("figure/neuralnetTrainLossVsLrDigit.pdf", width = 11, height = 11 * 0.25)
    tmp = lapply(modelViaDiffLr, function(x) x$metrics$loss)
    par(mar = c(4.2, 5, 0, 0), family = "serif", mfrow = c(1, 4))
    for(i in 1:length(tmp))
    {
      if(i == 1) xlab = "Epoch" else xlab = ""
      if(i == 1) ylab = "Training loss" else ylab = ""
      plot(tmp[[i]], type = "l", xlab = xlab, ylab = ylab, cex.lab = 2, cex.axis = 1.5, bty = "L", col = "gray20")
      if(i == 1) legend("top", legend = "Learning rate = 0.1", bty = "n", cex = 1.5)
      else if(i == 2) legend("top", legend = "0.01", bty = "n", cex = 1.5)
      else if(i == 3) legend("top", legend = "0.001", bty = "n", cex = 1.5)
      else if(i == 4) legend("top", legend = "0.0001", bty = "n", cex = 1.5)
    }
    dev.off()
    
    
  }
  
  
  
  
  # Cross validation with 2 models:
  # 1 hidden-8, dropout = 0.5, 1 hidden-4, dropout = 0.5, val_acc = 0.6308, 2000 epocs.
  # 1 hidden-128, dropout = 0.5, 1 hidden-64, dropout = 0.5, val_acc = 0.669: 1000 epocs.
  if(T)
  {
    
    
    model1cv = list()
    tmpX = as.matrix(dat[, -ncol(dat)])
    for(i in 1:length(valiIndList))
    {
      
      
      trainX = tmpX[-valiIndList[[i]], ]
      trainY = dat$y[-valiIndList[[i]]]
      validX = tmpX[valiIndList[[i]], ]
      validY = dat$y[valiIndList[[i]]]
      
      
      model = keras::keras_model_sequential()
      keras::layer_flatten(model, input_shape = ncol(trainX))
      
      
      keras::layer_dense(model, units = 8, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0)
      
      
      keras::layer_dense(model, units = 4, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0)
      
      
      keras::layer_dense(model, 10, activation = "softmax")
      
      
      opt = keras::optimizer_adam(lr = 0.0005)
      keras::compile(model, loss = "sparse_categorical_crossentropy", optimizer = opt, metrics = "accuracy")
      system.time({fm = keras::fit(model, x = trainX, y = trainY, batch_size = 32, epochs = 200, validation_data = list(validX, validY), verbose = 2, shuffle = T)}) # https://keras.rstudio.com/reference/fit.html
      model1cv[[i]] = fm
    }
    save(model1cv, file = "data/model1cvDigit.Rdata")
    
    
    
    
    model2cv = list()
    tmpX = as.matrix(dat[, -ncol(dat)])
    for(i in 1:length(valiIndList))
    {
      
      
      trainX = tmpX[-valiIndList[[i]], ]
      trainY = dat$y[-valiIndList[[i]]]
      validX = tmpX[valiIndList[[i]], ]
      validY = dat$y[valiIndList[[i]]]
      
      
      # tryCatch(keras::use_session_with_seed(42, disable_gpu = T, disable_parallel_cpu = F, quiet = F))
      model = keras::keras_model_sequential()
      keras::layer_flatten(model, input_shape = ncol(trainX))
      
      
      keras::layer_dense(model, units = 128, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0)
      
      
      keras::layer_dense(model, units = 64, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0)
      
      
      keras::layer_dense(model, 10, activation = "softmax")
      
      
      opt = keras::optimizer_adam(lr = 0.0005)
      keras::compile(model, loss = "sparse_categorical_crossentropy", optimizer = opt, metrics = "accuracy")
      system.time({fm = keras::fit(model, x = trainX, y = trainY, batch_size = 32, epochs = 200, validation_data = list(validX, validY), verbose = 2, shuffle = T)}) # https://keras.rstudio.com/reference/fit.html
      model2cv[[i]] = fm
    }
    save(model2cv, file = "data/model2cvDigit.Rdata")
    
    
    
    
    model3cv = list()
    tmpX = as.matrix(dat[, -ncol(dat)])
    for(i in 1:length(valiIndList))
    {
      
      
      trainX = tmpX[-valiIndList[[i]], ]
      trainY = dat$y[-valiIndList[[i]]]
      validX = tmpX[valiIndList[[i]], ]
      validY = dat$y[valiIndList[[i]]]
      
      
      # tryCatch(keras::use_session_with_seed(42, disable_gpu = T, disable_parallel_cpu = F, quiet = F))
      model = keras::keras_model_sequential()
      keras::layer_flatten(model, input_shape = ncol(trainX))
      
      
      keras::layer_dense(model, units = 128, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0.5)
      
      
      keras::layer_dense(model, units = 64, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0.5)
      
      
      keras::layer_dense(model, 10, activation = "softmax")
      
      
      opt = keras::optimizer_adam(lr = 0.0005)
      keras::compile(model, loss = "sparse_categorical_crossentropy", optimizer = opt, metrics = "accuracy")
      system.time({fm = keras::fit(model, x = trainX, y = trainY, batch_size = 32, epochs = 200, validation_data = list(validX, validY), verbose = 2, shuffle = T)}) # https://keras.rstudio.com/reference/fit.html
      model3cv[[i]] = fm
    }
    save(model3cv, file = "data/model3cvDigit.Rdata")
    
    
  }
  
  
  # Plot cross validation learning curves against epochs.
  if(T)
  {
    
    
    model1cvMean = lapply(1:length(model1cv[[1]]$metrics), function(i) rowMeans(as.data.frame(lapply(model1cv, function(x) x$metrics[[i]]))) )
    names(model1cvMean)= c("trainLoss", "trainErr", "validLoss", "validErr")
    model1cvMean$trainErr = 1 - model1cvMean$trainErr
    model1cvMean$validErr = 1 - model1cvMean$validErr
    
    
    model2cvMean = lapply(1:length(model2cv[[1]]$metrics), function(i) rowMeans(as.data.frame(lapply(model2cv, function(x) x$metrics[[i]]))) )
    names(model2cvMean)= c("trainLoss", "trainErr", "validLoss", "validErr")
    model2cvMean$trainErr = 1 - model2cvMean$trainErr
    model2cvMean$validErr = 1 - model2cvMean$validErr
    
    
    model3cvMean = lapply(1:length(model3cv[[1]]$metrics), function(i) rowMeans(as.data.frame(lapply(model3cv, function(x) x$metrics[[i]]))) )
    names(model3cvMean)= c("trainLoss", "trainErr", "validLoss", "validErr")
    model3cvMean$trainErr = 1 - model3cvMean$trainErr
    model3cvMean$validErr = 1 - model3cvMean$validErr
    
    
    # Plot neuralnetDigit
    if(T)
    {
      pdf("figure/neuralnetDigit.pdf", width = 9, height = 9 * 0.5)
      par(mar = c(4.2, 5, 0, 0), mfrow = c(2, 3), family = "serif")
      lossylim = range(c(range(model1cvMean$trainLoss), range(model1cvMean$validLoss), range(model2cvMean$trainLoss), range(model2cvMean$validLoss), range(model3cvMean$trainLoss), range(model3cvMean$validLoss)))
      
      
      plot(model1cvMean$trainLoss, ylim = lossylim, type = "l", col = "red", bty = "L", cex.lab = 2, cex.axis = 1.5, xlab = "", ylab = "Loss")
      lines(model1cvMean$validLoss, type = "l", col = "darkblue")
      legend("top", legend = "Model 1", bty = "n", cex = 1.5)
      legend("right", legend = c("Train mean", "CV mean"), bty = "n", cex = 1.5, col = c("red", "darkblue"), pch = c(15, 15))
      
      
      plot(model2cvMean$trainLoss, ylim = lossylim, type = "l", col = "red", bty = "L", cex.lab = 2, cex.axis = 1.5, xlab = "", ylab = "")
      lines(model2cvMean$validLoss, type = "l", col = "darkblue")
      legend("top", legend = "Model 2", bty = "n", cex = 1.5)
      
      
      plot(model3cvMean$trainLoss, ylim = lossylim, type = "l", col = "red", bty = "L", cex.lab = 2, cex.axis = 1.5, xlab = "", ylab = "")
      lines(model3cvMean$validLoss, type = "l", col = "darkblue")
      legend("top", legend = "Model 3", bty = "n", cex = 1.5)
      
      
      
      
      errylim = range(c(range(model1cvMean$trainErr), range(model1cvMean$validErr), range(model2cvMean$trainErr), range(model2cvMean$validErr), range(model3cvMean$trainErr), range(model3cvMean$validErr)))
      
      
      plot(model1cvMean$trainErr, ylim = errylim, type = "l", col = "skyblue", bty = "L", ylab = "Classification error", xlab = "Epoch", cex.lab = 2, cex.axis =1.5)
      lines(model1cvMean$validErr, type = "l", col = "olivedrab3")
      # legend("top", legend = "Model 1", bty = "n", cex = 1.5)
      legend("topright", legend = c("CV mean", "Train mean"), bty = "n", cex = 1.5, col = c("olivedrab3", "skyblue"), pch = c(15, 15))
      legend("right", legend = "min(CV) = 0.0636", bty = "n", cex = 1.5)
      
      
      plot(model2cvMean$trainErr, ylim = errylim, type = "l", col = "skyblue", bty = "L", xlab = "", ylab = "", cex.lab = 2, cex.axis = 1.5)
      lines(model2cvMean$validErr, type = "l", col = "olivedrab3")
      # legend("top", legend = "Model 2", bty = "n", cex = 1.5)
      legend("right", legend = "min(CV) = 0.0161", bty = "n", cex = 1.5)
      
      
      plot(model3cvMean$trainErr, ylim = errylim, type = "l", col = "skyblue", bty = "L", xlab = "", ylab = "", cex.lab = 2, cex.axis = 1.5)
      lines(model3cvMean$validErr, type = "l", col = "olivedrab3")
      # legend("top", legend = "Model 3", bty = "n", cex = 1.5)
      legend("right", legend = "min(CV) = 0.0133", bty = "n", cex = 1.5)
      
      
      dev.off()  
    }
    
    
  }
  
  
  # Selected # 1 hidden-128, dropout = 0.5, 1 hidden-64, dropout = 0.5, val_acc = 0.669: 1000 epocs.
  # Plot learning curve against training size.
  if(T)
  {
    tmpX = as.matrix(dat[, -ncol(dat)])
    # tmpX = apply(tmpX, 2, function(x) (x - min(x)) / max(x))
    # tmpY = as.integer((dat[[ncol(dat)]] + 1L) / 2L)
    tmpY = dat$y
    
    
    modelsForTrainSizes = list()
    for(i in 1:length(pieceList))
    {
      cat(i, "")
      trainX = tmpX[pieceList[[i]], ]
      trainY = tmpY[pieceList[[i]]]
      
      
      model = keras::keras_model_sequential()
      keras::layer_flatten(model, input_shape = ncol(trainX))
      
      
      keras::layer_dense(model, units = 128, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0.5)
      
      
      keras::layer_dense(model, units = 64, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0.5)
      
      
      keras::layer_dense(model, units = 10, activation = "softmax")
      
      
      opt = keras::optimizer_adam(lr = 0.0005)
      keras::compile(model, loss = "sparse_categorical_crossentropy", optimizer = opt, metrics = "accuracy")
      system.time({fm = keras::fit(model, x = trainX, y = trainY, batch_size = 32, epochs = 200, validation_split = 0.3, verbose = 0, shuffle = T)}) # https://keras.rstudio.com/reference/fit.html
      modelsForTrainSizes[[i]] = fm
    }
    
    
    # trainErr = unlist(lapply(modelsForTrainSizes, function(x) 1 - mean(tail(x$metrics$accuracy, n = 30))))
    # validErr = unlist(lapply(modelsForTrainSizes, function(x) 1 - mean(tail(x$metrics$val_accuracy, n = 30))))
    trainErr = unlist(lapply(modelsForTrainSizes, function(x) 1 - mean(x$metrics$accuracy[80:90])))
    validErr = unlist(lapply(modelsForTrainSizes, function(x) 1 - mean(x$metrics$val_accuracy[80:90])))
    pdf("figure/neuralnetDigitLearningCurveAgainstDataSize.pdf", width = 9, height = 9 * 0.618)
    par(mar = c(4.2, 5, 0, 0), family = "serif")
    plot(y = trainErr, x = dataSizes,  col = "red", ylim = range(c(trainErr, validErr)), type = "l", xlab = "N(observation)", ylab = "Error", bty = "L", cex.axis = 1.5, cex.lab = 2, lwd = 2)
    lines(y = validErr,  x = dataSizes, col = "darkblue", lwd  =2)
    legend("right", legend = c("Cross validation error", "Training error"), lwd = c(2, 2), col = c("darkblue", "red"), bty = "n", cex = 2)
    dev.off()
    
    
    save(modelsForTrainSizes, file = "data/modelsForTrainSizesANN.Rdata")
    
    # Boy, you finished editing the code. Just run it.
    
  }
  
}




















































