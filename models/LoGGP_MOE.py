# Copyright (c) by Alejandro Ordonez Conejo under BSD License
# Last modified: Armin Lederer 02/2022

import numpy as np
import random


class LoGGP_MOE:
    def __init__(self, xSize, outs, pts, N, ard, pretrained_params=None):
        # user defined values
        self.pts = pts  # data limit per local GP
        self.N = N  # max. number of leaves
        self.xSize = xSize  # dimensionality of X
        self.outs = outs  # dimensionality of Y
        # parameters
        self.wo = 300  # factor width / overlapping
        self.sigmaF = np.ones(outs, dtype=float)
        self.sigmaN = np.ones(outs, dtype=float)
        if ard:
            self.lengthS = np.ones([xSize, outs], dtype=float)
            self.amountL = xSize
        else:
            self.lengthS = 1
            self.amountL = 1
        # data
        self.X = np.zeros([xSize, pts * N], dtype=float)
        self.Y = np.zeros([outs, pts * N], dtype=float)
        self.K = np.zeros([pts * outs, pts * N], dtype=float)  # covariance matrix
        self.alpha = np.zeros([pts * outs, N], dtype=float)  # prediction vectors
        # aux variables
        self.count = 0  # number of leaves - 1
        self.localCount = np.zeros(2 * N - 1, dtype=int)  # pts in each set
        self.medians = np.zeros(2 * N - 1, dtype=float)  # vector of hyperplanes
        self.overlapD = np.zeros(2 * N - 1, dtype=int)  # overlap dimension
        self.overlapW = np.zeros(2 * N - 1, dtype=float)  # overlap width
        self.auxUbic = np.zeros(2 * N - 1, dtype=int) - 1  # map the position of data
        self.auxUbic[0] = 0
        self.children = np.zeros([2, 2 * N - 1], dtype=int) - 1  # line 1: left child, line 2: right child
        
        self.pretrained_params = pretrained_params
        
        self.L = np.zeros([outs, pts, pts * N], dtype=float)  # Cholesky factorization of K
        self.auxAlpha = np.zeros([pts * outs, N], dtype=float)  # aux alpha for Cholesky
        
        if self.pretrained_params is not None:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!Using pretrained parameters for LoGGP.")
            outputscale, noise, lengthscale = self.pretrained_params
            print(f"Outputscale: {outputscale}, Noise: {noise}, Lengthscale: {lengthscale}")
            self.sigmaF = np.atleast_1d(outputscale).flatten()
            self.sigmaN = np.atleast_1d(noise).flatten()
            if lengthscale.ndim == 1:
                self.lengthS = np.tile(np.atleast_1d(lengthscale).flatten(), (self.xSize, 1))
            elif lengthscale.ndim == 2:
                if lengthscale.shape[0] == self.xSize:
                    self.lengthS = lengthscale
                elif lengthscale.shape[1] == outs:
                    self.lengthS = np.tile(lengthscale, (1, outs))
                else:
                    raise ValueError(f"Lengthscale shape mismatch: got {lengthscale.T.shape}, expected ({self.xSize}, {outs})")
            else:
                raise ValueError(f"Lengthscale must be 1D or 2D, got {lengthscale.ndim}D")

    def kernel(self, Xi, Xj, out):
        if Xi.ndim == 1:
            kern = (self.sigmaF[out] ** 2) * np.exp(-0.5 * np.sum(((Xi - Xj) / self.lengthS.T[:, out]) ** 2))
            return kern
        else:
            kern = (self.sigmaF[out] ** 2) * np.exp(
                -0.5 * np.sum(((Xi.transpose() - Xj) / self.lengthS.T[:, out]) ** 2, axis=1))
            return kern

    # def updateParam(self, x, model):
    #     self.localCount[model] += 1
    #     pos = self.auxUbic[model]
    #     if self.localCount[model] == 1:
    #         for p in range(self.outs):
    #             kVal = self.kernel(x, x, p)
    #             self.K[p * self.pts, pos * self.pts] = kVal + self.sigmaN[p] ** 2
    #             self.alpha[p * self.pts, pos] = self.Y[p, pos * self.pts] / kVal
    #     else:
    #         auxX = self.X[:, pos * self.pts: pos * self.pts + self.localCount[model]]  # does not include x
    #         auxY = self.Y[:, pos * self.pts: pos * self.pts + self.localCount[model]]
    #         for p in range(self.outs):
    #             b = self.kernel(auxX, x, p)
    #             b[-1] += self.sigmaN[p] ** 2
    #             auxOut = p * self.pts

    #             self.K[auxOut + self.localCount[model] - 1,
    #             pos * self.pts: pos * self.pts + self.localCount[model] - 1] = b[0:-1]
    #             self.K[auxOut: auxOut + self.localCount[model],
    #             pos * self.pts + self.localCount[model] - 1] = np.transpose(b)

    #             self.alpha[auxOut: auxOut + self.localCount[model], pos] = \
    #                 np.linalg.solve(self.K[auxOut: auxOut + self.localCount[model],
    #                                 pos * self.pts: pos * self.pts + self.localCount[model]], auxY[p, :])
    
    def updateParam(self, x, model):
        self.localCount[model] += 1
        pos = self.auxUbic[model]
        if self.localCount[model] == 1:
            for p in range(self.outs):
                kVal = self.kernel(x, x, p)
                self.K[p * self.pts, pos * self.pts] = kVal + self.sigmaN[p] ** 2
                self.alpha[p * self.pts, pos] = self.Y[p, pos * self.pts] / kVal
        else:
            auxX = self.X[:, pos * self.pts: pos * self.pts + self.localCount[model]]  # does not include x
            auxY = self.Y[:, pos * self.pts: pos * self.pts + self.localCount[model]]
            for p in range(self.outs):
                b = self.kernel(auxX, x, p)
                b[-1] += self.sigmaN[p] ** 2
                auxOut = p * self.pts

                self.K[auxOut + self.localCount[model] - 1,
                pos * self.pts: pos * self.pts + self.localCount[model] - 1] = b[0:-1]
                self.K[auxOut: auxOut + self.localCount[model],
                pos * self.pts + self.localCount[model] - 1] = np.transpose(b)

                self.alpha[auxOut: auxOut + self.localCount[model], pos] = \
                    np.linalg.solve(self.K[auxOut: auxOut + self.localCount[model],
                                    pos * self.pts: pos * self.pts + self.localCount[model]], auxY[p, :])



    def addPoint(self, x, y, model):
        if self.localCount[model] < self.pts:
            self.X[:, self.auxUbic[model] * self.pts + self.localCount[model]] = x
            self.Y[:, self.auxUbic[model] * self.pts + self.localCount[model]] = y
            self.updateParam(x, model)
        if self.localCount[model] == self.pts:
            self.divide(model)

    def divide(self, model):
        if self.auxUbic[-1] != -1:
            # print("no room for more divisions")
            return
        # compute widths in all dimensions
        width = self.X[:, self.auxUbic[model] * self.pts: self.auxUbic[model] *
                                                          self.pts + self.pts].max(axis=1) - self.X[:,
                                                                                             self.auxUbic[model] *
                                                                                             self.pts: self.auxUbic[
                                                                                                           model] * self.pts + self.pts].min(
            axis=1)

        # obtain cutting dimension
        cutD = np.argmax(width)
        width = width.max()
        # compute hyperplane
        mP = (self.X[cutD, self.auxUbic[model] * self.pts: self.auxUbic[model] *
                                                           self.pts + self.pts].max() + self.X[cutD,
                                                                                        self.auxUbic[model] *
                                                                                        self.pts: self.auxUbic[
                                                                                                      model] * self.pts + self.pts].min()) / 2

        # get overlapping region
        o = width / self.wo
        if o == 0:
            o = 0.1

        self.medians[model] = mP  # set model hyperplane
        self.overlapD[model] = cutD  # cut dimension
        self.overlapW[model] = o  # width of overlap

        xL = np.zeros([self.xSize, self.pts], dtype=float)
        xR = np.zeros([self.xSize, self.pts], dtype=float)
        yL = np.zeros([self.outs, self.pts], dtype=float)
        yR = np.zeros([self.outs, self.pts], dtype=float)

        lcount = 0
        rcount = 0

        iL = np.zeros(self.pts, dtype=int)
        iR = np.zeros(self.pts, dtype=int)

        for i in range(self.pts):
            xD = self.X[cutD, self.auxUbic[model] * self.pts + i]
            if xD < mP - 0.5 * o:
                xL[:, lcount] = self.X[:, self.auxUbic[model] * self.pts + i]
                yL[:, lcount] = self.Y[:, self.auxUbic[model] * self.pts + i]
                iL[lcount] = i
                lcount += 1
            elif xD >= mP - 0.5 * o and xD <= mP + 0.5 * o:  # if in overlapping
                pL = 0.5 + (xD - mP) / o  # prob. of being in left
                if pL >= random.random() and pL != 0:  # left selected
                    xL[:, lcount] = self.X[:, self.auxUbic[model] * self.pts + i]
                    yL[:, lcount] = self.Y[:, self.auxUbic[model] * self.pts + i]
                    iL[lcount] = i
                    lcount += 1
                else:
                    xR[:, rcount] = self.X[:, self.auxUbic[model] * self.pts + i]
                    yR[:, rcount] = self.Y[:, self.auxUbic[model] * self.pts + i]
                    iR[rcount] = i
                    rcount += 1
            elif xD > mP + 0.5 * o:  # if in right
                xR[:, rcount] = self.X[:, self.auxUbic[model] * self.pts + i]
                yR[:, rcount] = self.Y[:, self.auxUbic[model] * self.pts + i]
                iR[rcount] = i
                rcount += 1
        self.localCount[model] = 0
        # update counter
        if self.count == 0:
            self.count += 1
        else:
            self.count += 2
        # assign children
        self.children[0, model] = self.count
        self.children[1, model] = self.count + 1
        # set local count of children
        self.localCount[self.count] = lcount
        self.localCount[self.count + 1] = rcount
        self.auxUbic[self.count] = self.auxUbic[model]
        self.auxUbic[self.count + 1] = self.auxUbic.max() + 1
        # values for K permutation
        order = np.concatenate((iL[0:lcount], iR[0:rcount]))
        # update parameters of child models
        for p in range(self.outs):
            newK = self.K[p * self.pts: (p + 1) * self.pts, self.auxUbic[model] * self.pts:
                                                            self.auxUbic[model] * self.pts + self.pts]
            # permute K
            newK[range(self.pts), :] = newK[order, :]
            newK[:, range(self.pts)] = newK[:, order]
            # set child K
            self.K[p * self.pts: p * self.pts + lcount, self.auxUbic[self.count] * self.pts:
                                                        self.auxUbic[self.count] * self.pts + lcount] = newK[0: lcount,
                                                                                                        0: lcount]
            self.K[p * self.pts: p * self.pts + rcount, self.auxUbic[self.count + 1] * self.pts:
                                                        self.auxUbic[self.count + 1] * self.pts + rcount] = \
                newK[lcount: self.pts, lcount: self.pts]
            # set child alpha
            self.alpha[p * self.pts: p * self.pts + lcount, self.auxUbic[self.count]] = \
                np.linalg.solve(newK[0: lcount, 0: lcount], yL[p, 0:lcount].transpose())
            self.alpha[p * self.pts: p * self.pts + rcount, self.auxUbic[self.count + 1]] = \
                np.linalg.solve(newK[lcount: self.pts, lcount: self.pts], yR[p, 0:rcount].transpose())
        # parent will not have more data:
        self.auxUbic[model] = -1
        # relocate X Y to children
        self.X[:, self.auxUbic[self.count] * self.pts:
                  self.auxUbic[self.count] * self.pts + self.pts] = xL
        self.X[:, self.auxUbic[self.count + 1] * self.pts:
                  self.auxUbic[self.count + 1] * self.pts + self.pts] = xR
        self.Y[:, self.auxUbic[self.count] * self.pts:
                  self.auxUbic[self.count] * self.pts + self.pts] = yL
        self.Y[:, self.auxUbic[self.count + 1] * self.pts:
                  self.auxUbic[self.count + 1] * self.pts + self.pts] = yR


    def activation(self, x, model):
        if (self.children[0, model] == -1):  # return 0 if model is a leaf
            return 0
        mP = self.medians[model]  # hyperplane value
        xD = x[self.overlapD[model]]  # x value in cut dimension
        o = self.overlapW[model]  # overlap width
        if xD < mP - 0.5 * o:
            return 1
        elif xD >= mP - 0.5 * o and xD <= mP + 0.5 * o:
            return 0.5 + (xD - mP) / o
        else:
            return 0

    def update(self, x, y):
        model = 0
        while self.children[0, model] != -1:  # while model is parent
            # search for a leaf
            pL = self.activation(x, model)
            if pL >= random.random() and pL != 0:
                model = self.children[0, model]  # go left
            else:
                model = self.children[1, model]  # go right
        self.addPoint(x, y, model)
        
    # def predict(self, x):
    #         x = np.asarray(x).flatten()
    #         out = np.zeros(self.outs, dtype=float)
    #         var_out = np.zeros(self.outs, dtype=float)

    #         models = np.zeros(1000, dtype=int)
    #         probs = np.ones(1000, dtype=float)

    #         mCount = 1
    #         while self.children[0, models[0:mCount]].sum() != -1 * mCount:
    #             for j in range(mCount):
    #                 if self.children[0, models[j]] != -1:
    #                     pL = self.activation(x, models[j])
    #                     if pL == 1:
    #                         models[j] = self.children[0, models[j]]
    #                     elif pL == 0:
    #                         models[j] = self.children[1, models[j]]
    #                     elif 0 < pL < 1:
    #                         models[mCount] = self.children[1, models[j]]
    #                         probs[mCount] = probs[j] * (1 - pL)
    #                         models[j] = self.children[0, models[j]]
    #                         probs[j] *= pL
    #                         mCount += 1

    #         for p in range(self.outs):
    #             pred_sum = 0
    #             pred_sq_sum = 0
    #             for i in range(mCount):
    #                 model = models[i]
    #                 if self.localCount[model] == 0:
    #                     continue
    #                 X_model = self.X[:, self.auxUbic[model] * self.pts: self.auxUbic[model] * self.pts + self.localCount[model]]
    #                 alpha_model = self.alpha[p * self.pts: p * self.pts + self.localCount[model], self.auxUbic[model]]
    #                 L_model = self.L[p, :self.localCount[model], self.auxUbic[model] * self.pts: self.auxUbic[model] * self.pts + self.localCount[model]]

    #                 k_star = self.kernel(X_model, x, p)
    #                 mu_i = np.dot(k_star, alpha_model)

                
    #                 try:
    #                     jitter = 1e-6
    #                     L_reg = L_model + jitter * np.eye(L_model.shape[0])
    #                     v = np.linalg.solve(L_reg, k_star)
    #                 except np.linalg.LinAlgError:
    #                     print(f"[PREDICT] Still failed after regularization, fallback to zeros")
    #                     v = np.zeros_like(k_star)

    #                 sigma_f_sq = self.sigmaF[p] ** 2
    #                 var_i = sigma_f_sq - np.dot(v, v)
    #                 var_i = max(var_i, 1e-6)

    #                 pred_sum += probs[i] * mu_i
    #                 pred_sq_sum += probs[i] * (mu_i ** 2 + var_i)

    #             out[p] = pred_sum
    #             var_out[p] = max(pred_sq_sum - pred_sum ** 2, 1e-6)

    #         return out, var_out
    
    def predict(self, x):
        x = np.asarray(x).flatten()
        out = np.zeros(self.outs, dtype=float)
        var_out = np.zeros(self.outs, dtype=float)

        models = np.zeros(1000, dtype=int)
        probs = np.ones(1000, dtype=float)

        mCount = 1
        while self.children[0, models[0:mCount]].sum() != -1 * mCount:
            for j in range(mCount):
                if self.children[0, models[j]] != -1:
                    pL = self.activation(x, models[j])
                    if pL == 1:
                        models[j] = self.children[0, models[j]]
                    elif pL == 0:
                        models[j] = self.children[1, models[j]]
                    elif 0 < pL < 1:
                        models[mCount] = self.children[1, models[j]]
                        probs[mCount] = probs[j] * (1 - pL)
                        models[j] = self.children[0, models[j]]
                        probs[j] *= pL
                        mCount += 1

        for p in range(self.outs):
            pred_sum = 0.0
            pred_sq_sum = 0.0

            for i in range(mCount):
                model = models[i]
                weight = probs[i]
                pos = self.auxUbic[model]
                if pos == -1 or self.localCount[model] == 0:
                    continue

                X_train = self.X[:, pos * self.pts : pos * self.pts + self.localCount[model]]
                y_alpha = self.alpha[p * self.pts : p * self.pts + self.localCount[model], pos]

                k_star = self.kernel(X_train, x, p)
                pred_mean = np.dot(k_star, y_alpha)

                try:
                    K_sub = self.K[p * self.pts : p * self.pts + self.localCount[model],
                                pos * self.pts : pos * self.pts + self.localCount[model]]
                    v = np.linalg.solve(K_sub, k_star)
                    pred_var = self.kernel(x, x, p) - np.dot(k_star, v)
                    pred_var = max(pred_var, 1e-6)
                except np.linalg.LinAlgError:
                    pred_var = 1e-6  # fallback for numerical issues

                pred_sum += weight * pred_mean
                pred_sq_sum += weight * (pred_mean ** 2 + pred_var)

            out[p] = pred_sum
            var_out[p] = max(pred_sq_sum - pred_sum ** 2, 1e-6)

        return out, var_out
