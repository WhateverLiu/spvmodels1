

source("r/rfuns.r")
tmp = data.table::setDF(data.table::fread("data/madelon_train.data", header = F))
tmp2 = data.table::setDF(data.table::fread("data/madelon_valid.data", header = F))
dat = rbind(tmp, tmp2)
tmp = data.table::setDF(data.table::fread("data/madelon_train.labels", header = F))
tmp2 = data.table::setDF(data.table::fread("data/madelon_valid.labels", header = F))
dat$y = c(tmp[[1]], tmp2[[1]])
set.seed(123)
dat = dat[sample(nrow(dat)), ]


cvNfold = 10L
valiIndList = generateChunkInd(nrow(dat), cvNfold)
pieceList = generateTrainDataSeq(nrow(dat), Npiece = 20)
dataSizes = unlist(lapply(pieceList, function(x) length(x)))


# Feature selection via variable importance in decision tree and PCA.
if(T)
{
  
  
  fml = paste0(colnames(dat)[ncol(dat)], "~", paste0(colnames(dat)[-ncol(dat)], collapse = "+"))
  fml = eval(parse(text = fml))
  tmptree = rpart::rpart(fml, data = dat, method = "class", control = rpart::rpart.control(minsplit = 1, cp = 0, xval = 0))
  tmp = tmptree$variable.importance
  
  
  pdf("figure/madelonVariableImportance.pdf", width = 9, height = 9 * 0.618)
  par(mar = c(4.2, 5, 0, 0), family = "serif")
  plot(tmp, type = "l", cex.lab = 2, cex.axis = 1.5, xlab = "Variable name", ylab = "Variable importance", bty = "L", xaxt = "n")
  lines(tmp, type = "p", col = "blue")
  axis(side = 1, at = 1:length(tmp), labels = names(tmp), cex = 0.5)
  lines(x = c(-1e10, 1e10), y = c(tmp[20], tmp[20]) * 0.9, lty = 2)
  dev.off()
  
  
  dat = dat[c(names(tmp)[1:20], "y")]
  tmp = svd(as.matrix(apply(dat[-ncol(dat)], 2, function(x) x - mean(x))))
  
  
  pdf("figure/madelonVariableImportanceEigen.pdf", width = 9, height = 9 * 0.618)
  par(mar = c(4.2, 5, 0, 0), family = "serif")
  plot(log10(tmp$d), bty = "L", cex.lab = 2, cex.axis = 1.5, xlab = "Singular value index", ylab = "Singular value", yaxt = "n")
  # plot((tmp$d), bty = "L", cex.lab = 2, cex.axis = 1.5, xlab = "Singular value index", ylab = "Singular value", yaxt = "n")
  axis(side = 2, at = log10(c(1, 10, 100, 1000, 1e4, 1e5)), labels = c(1, 10, 100, 1000, 1e4, 1e5), cex.axis = 1.5)
  lines(x = c(-1e10, 1e10), y = numeric(2) + log10(tmp$d[5]) * 0.9, lty = 2)
  dev.off()
  
  
  tmp = tmp$u[, 1:5] %*% diag(tmp$d[1:5])
  tmp = apply(tmp, 2, function(x) x - min(x))
  tmpmax = max(tmp)
  tmp = apply(tmp, 2, function(x) x / tmpmax)
  dat = data.frame(tmp, y = dat$y)
  
  
}


# Single decision tree.
if(T)
{
  
  
  # See learning curve
  fml = paste0(colnames(dat)[ncol(dat)], "~", paste0(colnames(dat)[-ncol(dat)], collapse = "+"))
  fml = eval(parse(text = fml))
  dtreeLearned = lapply(pieceList, function(x)
  {
    cat(".")
    alldat = dat[x, ]
    tmptree = rpart::rpart(fml, data = alldat, method = "class", control = rpart::rpart.control(minsplit = 5, cp = 1e-10, xval = 10))
    cp = tmptree$cptable[which.min(tmptree$cptable[, "xerror"]), "CP"]
    # finaltree = rpart::prune.rpart(tmptree, cp = cp)
    finaltree = tmptree
    valilist = generateChunkInd(nrow(alldat), Nchunk = 10)
    trainValidErr = as.data.frame(t(as.data.frame(lapply(valilist, function(u)
    {
      train = alldat[-u, ]
      valid = alldat[u, ]
      tmptree = rpart::rpart(fml, data = train, method = "class", control = rpart::rpart.control(minsplit = 5, cp = cp, xval = 0))
      trainPred = predict(tmptree, newdata = train, "class")
      trainErr = sum(trainPred != train[[ncol(train)]]) / nrow(train)
      validErr = sum(predict(tmptree, newdata = valid, "class") != valid[[ncol(valid)]]) / nrow(valid)
      c(trainErr, validErr)
    }))))
    rownames(trainValidErr) = NULL
    colnames(trainValidErr) = c("trainErr", "validErr")
    list(model = finaltree, cvErr = trainValidErr)
  })
  
  
  pdf("figure/dtreeAugmentedMadelonCp.pdf", width = 9, height = 9 * 0.618)
  par(mar = c(4.5, 5, 5, 0), family = "serif", cex.lab = 2, cex.axis = 1.5)
  rpart::plotcp(dtreeLearned[[length(dtreeLearned)]]$model)
  dev.off()
  
  
  tmp = as.data.frame(t(as.data.frame(lapply(dtreeLearned, function(x) colMeans(x$cvErr)))))
  rownames(tmp) = NULL; colnames(tmp) = c("trainErr", "validErr")
  
  
  pdf("figure/dtreeLearningCurveMadelonAugmented.pdf", width = 9, height = 9 * 0.618)
  par(mar = c(4, 5, 0, 0), family = "serif")
  plot(x = dataSizes, y = tmp$trainErr, ylim = range(unlist(tmp)), type = "l", col = "red", xlab = "N(observation)", ylab = "Error (misclassification rate)", bty = "L", cex.lab = 2, cex.axis = 1.5, lwd = 2)
  legend("topright", legend = c("Cross validation error", "Training error", "Full data CV error = 0.196"), col = c("darkblue", "red", "white"), cex = 2, lwd = c(2, 2, 1), bty = "n")
  lines(x = dataSizes, y = tmp$validErr, col = "darkblue", lwd = 2)
  dev.off()
  save(dtreeLearned, file = "data/dtreeLearingCurveMadelonAugmented.Rdata")
  
  
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
      rst[[i]] = xgboost::xgb.cv(params = list(objective = "binary:logistic", eta = eta[i], max_depth = 6, nthread = 13), data = tmpdat, nrounds = 100, metrics = "error", label = as.integer((dat[[ncol(dat)]] + 1L) / 2L), nfold = 10, verbose = F)
    }
    rm(tmpdat); gc()
    save(rst, file = "data/xgboostRstMadelonAugmented.Rdata")
    
    
    tmp = as.data.frame(lapply(rst, function(x) x$evaluation_log$test_error_mean))
    colnames(tmp) = paste0("lr", eta); rownames(tmp) = NULL
    optLrate = (as.integer(which.min(unlist(tmp)) / 100) + 1L) / 100
    optNtree = as.integer(which.min(unlist(tmp))) %% 100L
    tmp = t(tmp)
    minErr = min(tmp)
    maxErr = max(tmp)
    tmp = as.matrix(tmp[nrow(tmp):1, ])
    dimnames(tmp) = NULL
    tmp[order(tmp)] = qnorm(seq(0.005, 0.995, len = length(tmp)))
    pdf("figure/xgboostLrNtreeMadelonAugmented.pdf", width = 10, height = 8)
    par(mar = c(4, 5, 0, 0), family = "serif")
    image(t(tmp)[, nrow(tmp):1], col = unique(colorRampPalette(c("blue", "white", "red"))(20000)), xlab = "N(tree)", ylab = "Learning rate", cex.lab = 2, xaxt = "n", yaxt = "n", xlim = c(0, 1.2), bty = "n")
    axis(side = 1, at = seq(0, 1, len = 100), labels = 1:100, cex.axis = 1.5)
    axis(side = 2, at = seq(0, 1, len = length(eta)), labels = eta, cex.axis = 1.5)
    lines(x = rep(1.05, 100), y = seq(0.3, 0.7, len = 100), pch = 15, col = colorRampPalette(c("blue", "white", "red"))(100), type = "p")
    text(x = 1.11, y = 0.3, labels = paste0("Low\n", round(minErr, 3)), cex = 1.7)
    text(x = 1.11, y = 0.75, labels = paste0("High error\n", round(maxErr, 3)), cex = 1.7)
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
      tmp = xgboost::xgb.cv(params = list(objective = "binary:logistic", eta = optLrate, max_depth = 6, nthread = 13), data = as.matrix(alldat[-ncol(alldat)]), nrounds = optNtree, metrics = "error", label = as.integer((alldat[[ncol(alldat)]] + 1L) / 2L), nfold = 10, verbose = F)
      n = length(tmp$evaluation_log$test_error_mean)
      c(tmp$evaluation_log$train_error_mean[n], tmp$evaluation_log$test_error_mean[n])
    }))))
    rownames(trainValidErr) = NULL
    colnames(trainValidErr) = c("trainErr", "validErr")
    
    
    learningCurve = data.frame(dataSize = unlist(lapply(pieceList, function(x) length(x))), trainValidErr)
    tmp = learningCurve
    pdf("figure/xgboostLearningCurveMadelon.pdf", width = 9, height = 9 * 0.618)
    par(mar = c(4, 5, 0, 0), family = "serif")
    plot(x = tmp$dataSize, y = tmp$trainErr, ylim = range(unlist(tmp[-1])), type = "l", col = "red", xlab = "N(observation)", ylab = "Error", bty = "L", cex.lab = 2, cex.axis = 1.5, lwd = 2)
    legend("topright", legend = c("Cross validation error", "Training error"), col = c("darkblue", "red"), cex = 2, lwd = c(2, 2), bty = "n")
    lines(x = tmp$dataSize, y = tmp$validErr, col = "darkblue", lwd = 2)
    dev.off()
    save(learningCurve, file = "data/xgboostLearingCurveMadelon.Rdata")
    
    
  }
}





# Support vector machine.
if(T)
{
  
  
  X = as.matrix(dat[-ncol(dat)])
  # X = as.matrix(as.data.frame(lapply(X, function(x) (x - mean(x)) / sd(x))))
  X = apply(X, 2, function(x) x - mean(x))
  
  
  # Gaussian kernel.
  if(T)
  {
    
    
    rbfXs = lapply(c(100, 10, 1, 1e-1), function(s) kernlab::kernelMatrix(kernlab::rbfdot(sigma = s), X))
    lambda = unique(c(seq(0.01, 0.1, by = 0.01), c(seq(0.1, 1, by = 0.1), c(seq(1, 10, by = 1), seq(10, 100, by = 10)))))
    maxCore = length(lambda)
    computelist = lapply(lambda, function(x) list(lambda = x, kernXs = rbfXs, y = dat$y))
    cl = snow::makeCluster(maxCore, type = "SOCK")
    rbfKernelRst = snow::clusterApply(cl, computelist, function(X)
    {
      lambda = X$lambda; y = as.factor(X$y)
      lapply(X$kernXs, function(x)
      {
        set.seed(42)
        kernlab::ksvm(x = x, y = y, scaled = F, type = "C-svc", kernel = "matrix", C = lambda, cross = 10, cache = 2048)
      })
    })
    snow::stopCluster(cl) 
    save(rbfKernelRst, file = "data/rbfKernelRstMadelonAugmented.Rdata")
    
    
    pdf("figure/rbfKernelRstCvErrInitialMadelonAugmented.pdf", width = 9, height = 9 * 0.618)
    par(mar = c(4, 5, 0, 0), family = "serif")
    tmp = lapply(1:length(rbfKernelRst[[1]]), function(i) unlist(lapply(rbfKernelRst, function(x) x[[i]]@cross)))
    ylim = range(unlist(tmp)); xlim = c(1, length(tmp[[1]]))
    plot(0, xlim = xlim, ylim = ylim, col = "white", bty = "L", xlab = expression(lambda), ylab = "Cross validation error", xaxt = "n", cex.lab = 2, cex.axis = 1.5)
    axis(side = 1, at = 1:length(tmp[[1]]), labels = lambda, cex.axis = 1.5)
    linecols = c("black", "red", "blue", "olivedrab3")
    legend("right", legend = c(expression(sigma == 100), expression(sigma == 10), expression(sigma == 1), expression(sigma == 0.1)), col = linecols, pch = c(15, 15, 15, 15), cex = 1.5, bty = "n")
    for(i in 1:length(tmp))
    {
      lines(tmp[[i]], col = linecols[i], lwd = 2)
    }
    legend("topright", legend = paste0("min(err) = ", round(min(unlist(tmp)), 3)), cex = 1.5, bty = "n")
    dev.off()
    
    
  }
  
  
  
  
  # Laplacian kernel.
  if(T)
  {
    
    
    lapXs = lapply(c(10, 1, 1e-1, 1e-2), function(s) kernlab::kernelMatrix(kernlab::laplacedot(sigma = s), X))
    lambda = unique(c(seq(0.01, 0.1, by = 0.01), seq(0.1, 1, by = 0.1), seq(1, 10, by = 1), seq(10, 100, by = 10)))
    maxCore = length(lambda)
    computelist = lapply(lambda, function(x) list(lambda = x, kernXs = lapXs, y = dat$y))
    cl = snow::makeCluster(maxCore, type = "SOCK")
    lapKernelRst = snow::clusterApply(cl, computelist, function(X)
    {
      lambda = X$lambda; y = as.factor(X$y)
      lapply(X$kernXs, function(x)
      {
        set.seed(42)
        kernlab::ksvm(x = x, y = y, scaled = F, type = "C-svc", kernel = "matrix", C = lambda, cross = 10, cache = 2048)
      })
    })
    snow::stopCluster(cl) 
    save(lapKernelRst, file = "data/lapKernelRstMadelonAugmented.Rdata")
    
    
    pdf("figure/lapKernelRstCvErrInitialMadelonAugmented.pdf", width = 9, height = 9 * 0.618)
    par(mar = c(4, 5, 0, 0), family = "serif")
    tmp = lapply(1:length(lapKernelRst[[1]]), function(i) unlist(lapply(lapKernelRst, function(x) x[[i]]@cross)))
    ylim = range(unlist(tmp)); xlim = c(1, length(tmp[[1]]))
    plot(0, xlim = xlim, ylim = ylim, col = "white", bty = "L", xlab = expression(lambda), ylab = "Cross validation error", xaxt = "n", cex.lab = 2, cex.axis = 1.5)
    axis(side = 1, at = 1:length(tmp[[1]]), labels = lambda, cex.axis = 1.5)
    linecols = c("black", "red", "blue", "olivedrab3")
    legend("left", legend = c(expression(sigma == 10), expression(sigma == 1), expression(sigma == 0.1), expression(sigma == 0.01)), col = linecols, pch = c(15, 15, 15, 15), cex = 1.5, bty = "n")
    for(i in 1:length(tmp))
    {
      lines(tmp[[i]], col = linecols[i], lwd = 2)
    }
    legend("topright", legend = paste0("min(err) = ", round(min(unlist(tmp)), 3)), cex = 1.5, bty = "n")
    dev.off()
    
    
  }
  
  
  
  
  # min(err) = 0.125, opt(lambda) = 0.1, opt(sigma) = 100, Gaussian kernel.
  # Plot learning curve.
  if(T)
  {
    
    
    trainValidErr = as.data.frame(t(as.data.frame(lapply(pieceList, function(x)
    {
      cat(".")
      alldat = dat[x, ]
      lapX = kernlab::kernelMatrix(kernlab::rbfdot(sigma = 100), as.matrix(alldat[-ncol(alldat)]))
      set.seed(42)
      mdl = kernlab::ksvm(x = lapX, y = as.factor(alldat$y), scaled = F, type = "C-svc", kernel = "matrix", C = 0.1, cross = 10, cache = 2048)
      c(mdl@error, mdl@cross)
    }))))
    rownames(trainValidErr) = NULL
    colnames(trainValidErr) = c("trainErr", "validErr")
    
    
    learningCurve = data.frame(dataSize = unlist(lapply(pieceList, function(x) length(x))), trainValidErr)
    tmp = learningCurve
    pdf("figure/rbfKernelSVMlearningCurveMadelonAugmented.pdf", width = 9, height = 9 * 0.618)
    par(mar = c(4, 5, 0, 0), family = "serif")
    plot(x = tmp$dataSize, y = tmp$trainErr, ylim = range(unlist(tmp[-1])), type = "l", col = "red", xlab = "N(observation)", ylab = "Error", bty = "L", cex.lab = 2, cex.axis = 1.5, lwd = 2)
    legend("topright", legend = c("Cross validation error", "Training error"), col = c("darkblue", "red"), cex = 2, lwd = c(2, 2), bty = "n")
    lines(x = tmp$dataSize, y = tmp$validErr, col = "darkblue", lwd = 2)
    dev.off()
    save(learningCurve, file = "data/rbfKernelSVMlearingCurveMadelonAugmented.Rdata")
  }
  
  
}




# K-nearest neighbor
# opt(K) = 35, min(cverr) = 0.238
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
        tmp = as.integer(sign(cumsum(nnYs)))
        tmp[tmp == 0L] = 1L
        tmp
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
  pdf("figure/cvErrKNNaugmentedMadeLon.pdf", width = 9, height = 9 * 0.618)
  par(mar = c(4, 5, 0, 0), family = "serif")
  plot(0, col = "white", xlim = c(1, length(cvErr)), ylim = range(unlist(cvErr)), bty = "L", cex.lab = 2, cex.axis = 1.5, ylab = "Cross validation error", xlab = "K", xaxt = "n")
  axis(side = 1, at = 1:length(cvErr), labels = 1:length(cvErr), cex.axis = 1.5)
  for(i in 1:length(cvErr))
  {
    m = mean(cvErr[[i]]); s = sd(cvErr[[i]])
    lines(x = i, y = m, type = "p", col = "darkblue")
    lines(x = c(i, i), y = c(m - s, m + s))
  }
  legend("topright", legend = c("min(err) = 0.123", "opt(K) = 11"), bty = "n", cex = 2)
  dev.off()
  
  
  tmp = unlist(lapply(knnCVs, function(x) min(colMeans(x))))
  pdf("figure/KNNlearningCurveAugmentedMadelon.pdf", width = 9, height = 9 * 0.618)
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
    
    
    # i = 1L
    tmpX = as.matrix(dat[, -ncol(dat)])
    trainX = tmpX
    trainY = as.integer((dat$y + 1L) / 2L)
    
    
    # keras::use_session_with_seed(42, disable_gpu = F, disable_parallel_cpu = F, quiet = F)
    model = keras::keras_model_sequential()
    keras::layer_flatten(model, input_shape = ncol(trainX))
    
    
    keras::layer_dense(model, units = 64, activation = NULL) # "relu"
    keras::layer_activation_leaky_relu(model, alpha = 0.01)
    keras::layer_dropout(model, rate = 0)
    
    
    keras::layer_dense(model, units = 64, activation = NULL) # "relu"
    keras::layer_activation_leaky_relu(model, alpha = 0.01)
    keras::layer_dropout(model, rate = 0)
    
    
    # keras::layer_dense(model, units = 4, activation = NULL) # "relu"
    # keras::layer_activation_leaky_relu(model, alpha = 0.01)
    # keras::layer_dropout(model, rate = 0.5)
    
    
    keras::layer_dense(model, units = 1, activation = "sigmoid")
    # keras::layer_dense(model, units = 2, activation = "softmax")
    
    
    opt = keras::optimizer_adam(lr = 1e-2, decay = 1 - 0.999965876413334)
    keras::compile(model, 
                   loss = "binary_crossentropy",
                   # loss = "sparse_categorical_crossentropy", 
                   optimizer = opt, metrics = "accuracy")
    system.time({fm = keras::fit(
      model, x = trainX, y = trainY, batch_size = 32, epochs = 500, 
      validation_split = 0.3,
      # validation_data = list(validX, validY),
      verbose = 2, shuffle = T)}) # https://keras.rstudio.com/reference/fit.html
    gc()
    
    
    # 1 hidden layer of 4 neurons, no dropout, val_acc = 0.6115
    # 0 hidden layer, no dropout, val_acc = 0.5731
    # 1 hidden-8, dropout = 0.5, 1 hidden-4, dropout = 0.5, val_acc = 0.6308
    # 1 hidden-16, dropout = 0.5, 1 hidden-8, dropout = 0.5, 1 hidden-4, dropout = 0.5, val_acc = 0.5885
    # 1 hidden-128, dropout = 0.5, 1 hidden-64, dropout = 0.5, val_acc = 0.669: 250 epocs
  }
  
  
  # Find best learning rate, and plot.
  if(T)
  {
    
    
    tmpX = as.matrix(dat[, -ncol(dat)])
    tmpX = apply(tmpX, 2, function(x) (x - min(x)) / max(x))
    tmpY = as.integer((dat[[ncol(dat)]] + 1L) / 2L)
    
    
    lrlist = c(0.1, 0.01, 0.001, 0.0001)
    modelViaDiffLr = lapply(lrlist, function(r)
    {
      model = keras::keras_model_sequential()
      keras::layer_flatten(model, input_shape = ncol(tmpX))
      
      
      keras::layer_dense(model, units = 8, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0.5)
      
      
      keras::layer_dense(model, units = 4, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0.5)
      
      
      keras::layer_dense(model, units = 1, activation = "sigmoid")
      
      
      opt = keras::optimizer_adam(lr = r)
      keras::compile(model, loss = "binary_crossentropy", optimizer = opt, metrics = "accuracy")
      system.time({fm = keras::fit(model, x = tmpX, y = tmpY, batch_size = 32, epochs = 1000, validation_split = 0.3, verbose = 2, shuffle = T)}) # https://keras.rstudio.com/reference/fit.html
      fm
    })
    
    
    # Train-valid split 0.3
    pdf("figure/neuralnetTrainLossVsLrMadelon.pdf", width = 11, height = 11 * 0.25)
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
    # model1cv is ignored.
    
    
    model2cv = list()
    tmpX = as.matrix(dat[, -ncol(dat)])
    for(i in 1:length(valiIndList))
    {
      
      
      trainX = tmpX[-valiIndList[[i]], ]
      trainY = as.integer((dat$y[-valiIndList[[i]]] + 1L) / 2L)
      validX = tmpX[valiIndList[[i]], ]
      validY = as.integer((dat$y[valiIndList[[i]]] + 1L) / 2L)
      
      
      # tryCatch(keras::use_session_with_seed(42, disable_gpu = T, disable_parallel_cpu = F, quiet = F))
      model = keras::keras_model_sequential()
      keras::layer_flatten(model, input_shape = ncol(trainX))
      
      
      keras::layer_dense(model, units = 64, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0)
      
      
      keras::layer_dense(model, units = 64, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0)
      
      
      keras::layer_dense(model, units = 1, activation = "sigmoid")
      
      
      opt = keras::optimizer_adam(lr = 1e-2, decay = 1 - 0.999965876413334)
      keras::compile(model, loss = "binary_crossentropy", optimizer = opt, metrics = "accuracy")
      system.time({fm = keras::fit(model, x = trainX, y = trainY, batch_size = 32, epochs = 500, validation_data = list(validX, validY), verbose = 2, shuffle = T)}) # https://keras.rstudio.com/reference/fit.html
      model2cv[[i]] = fm
    }
    save(model2cv, file = "data/model2cvMadelonAugmented.Rdata")
    
 
  }
  
  
  # Plot cross validation learning curves against epochs.
  if(T)
  {
    
    
    # model1cvMean = lapply(1:length(model1cv[[1]]$metrics), function(i) rowMeans(as.data.frame(lapply(model1cv, function(x) x$metrics[[i]]))) )
    # names(model1cvMean)= c("trainLoss", "trainErr", "validLoss", "validErr")
    # model1cvMean$trainErr = 1 - model1cvMean$trainErr
    # model1cvMean$validErr = 1 - model1cvMean$validErr
    
    
    model2cvMean = lapply(1:length(model2cv[[1]]$metrics), function(i) rowMeans(as.data.frame(lapply(model2cv, function(x) x$metrics[[i]]))) )
    names(model2cvMean)= c("trainLoss", "trainErr", "validLoss", "validErr")
    model2cvMean$trainErr = 1 - model2cvMean$trainErr
    model2cvMean$validErr = 1 - model2cvMean$validErr
    
    
    # Plot neuralnetMadelon
    if(T)
    {
      
      
      pdf("figure/neuralnetMadelonAugmented.pdf", width = 8, height = 8 * 0.5)
      par(mar = c(4.2, 5, 0, 0), mfrow = c(1, 2), family = "serif")
      lossylim = range(c(range(model2cvMean$trainLoss), range(model2cvMean$validLoss)))
      
      
      plot(model2cvMean$trainLoss, ylim = lossylim, type = "l", col = "red", bty = "L", cex.lab = 2, cex.axis = 1.5, xlab = "Epoch", ylab = "Loss")
      lines(model2cvMean$validLoss, type = "l", col = "darkblue")
      legend("top", legend = "Model 2", bty = "n", cex = 1.5)
      legend("right", legend = c("   ", "   ", "Train mean", "CV"), col = c("white", "white", "red", "darkblue"), pch = c(15, 15, 15, 15), cex = 1.5, bty = "n", text.col = c("white", "white", "black", "black"))
      
      
      errylim = range(c(range(model2cvMean$trainErr), range(model2cvMean$validErr)))
      plot(model2cvMean$trainErr, ylim = errylim, type = "l", col = "skyblue", bty = "L", xlab = "", ylab = "", cex.lab = 2, cex.axis = 1.5)
      lines(model2cvMean$validErr, type = "l", col = "olivedrab3")
      legend("top", legend = "Model 2", bty = "n", cex = 1.5)
      legend("right", legend = c("Train mean", "CV", paste0("min(CV) = ", 0.117)), bty = "n", cex = 1.5, pch = c(15, 15, 15), col = c("skyblue", "olivedrab3", "white"))
      
      
      dev.off()  
    }
    
    
  }
  
  
  
  
  # Selected # 1 hidden-128, dropout = 0.5, 1 hidden-64, dropout = 0.5, val_acc = 0.669: 1000 epocs.
  # Plot learning curve against training size.
  if(T)
  {
    
    
    tmpX = as.matrix(dat[, -ncol(dat)])
    tmpY = as.integer((dat[[ncol(dat)]] + 1L) / 2L)
    
    
    modelsForTrainSizes = list()
    for(i in 1:length(pieceList))
    {
      cat(i, "")
      trainX = tmpX[pieceList[[i]], ]
      trainY = tmpY[pieceList[[i]]]
      
      
      model = keras::keras_model_sequential()
      keras::layer_flatten(model, input_shape = ncol(trainX))
      
      
      keras::layer_dense(model, units = 64, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0)
      
      
      keras::layer_dense(model, units = 64, activation = NULL) # "relu"
      keras::layer_activation_leaky_relu(model, alpha = 0.01)
      keras::layer_dropout(model, rate = 0)
      
      
      keras::layer_dense(model, units = 1, activation = "sigmoid")
      
      
      opt = keras::optimizer_adam(lr = 1e-2, decay = 1 - 0.999965876413334)
      keras::compile(model, loss = "binary_crossentropy", optimizer = opt, metrics = "accuracy")
      system.time({fm = keras::fit(model, x = trainX, y = trainY, batch_size = 32, epochs = 500, validation_split = 0.3, verbose = 0, shuffle = T)}) # https://keras.rstudio.com/reference/fit.html
      modelsForTrainSizes[[i]] = fm
    }
    
    
    trainErr = unlist(lapply(modelsForTrainSizes, function(x) 1 - max(cumsum(rev(x$metrics$accuracy)) / (1:length(x$metrics$accuracy)))))
    validErr = unlist(lapply(modelsForTrainSizes, function(x) 1 - max(cumsum(rev(x$metrics$val_accuracy)) / (1:length(x$metrics$val_accuracy)))))
    
    
    pdf("figure/neuralnetMadelonLearningCurveAgainstDataSizeAugmented.pdf", width = 9, height = 9 * 0.618)
    par(mar = c(4.2, 5, 0, 0), family = "serif")
    plot(y = trainErr, x = dataSizes,  col = "red", ylim = range(c(trainErr, validErr)), type = "l", xlab = "N(observation)", ylab = "Error", bty = "L", cex.axis = 1.5, cex.lab = 2, lwd = 2)
    lines(y = validErr,  x = dataSizes, col = "darkblue", lwd  =2)
    legend("topright", legend = c("Cross validation error", "Training error"), lwd = c(2, 2), col = c("darkblue", "red"), bty = "n", cex = 2)
    dev.off()
    
    
  }
  
}









































