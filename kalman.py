from __future__ import division
from __future__ import print_function
import numpy as np
from numpy.linalg import pinv, det

def smooth_update(xsmooth_future, Vsmooth_future, xfilt, Vfilt, Vfilt_future, VVfilt_future, A, Q, B, u):
    # One step of the backwards RTS smoothing equations.
    #
    # INPUTS:
    # xsmooth_future = E[X_t+1|T]
    # Vsmooth_future = Cov[X_t+1|T]
    # xfilt = E[X_t|t]
    # Vfilt = Cov[X_t|t]
    # Vfilt_future = Cov[X_t+1|t+1]
    # VVfilt_future = Cov[X_t+1,X_t|t+1]
    # A = system matrix for time t+1
    # Q = system covariance for time t+1
    # B = input matrix for time t+1 (or [] if none)
    # u = input vector for time t+1 (or [] if none)
    #
    # OUTPUTS:
    # xsmooth = E[X_t|T]
    # Vsmooth = Cov[X_t|T]
    # VVsmooth_future = Cov[X_t+1,X_t|T]

    #xpred = E[X(t+1) | t]
    if B.size == 0:
      xpred = A.dot(xfilt);
    else:
      xpred = A.dot(xfilt) + B.dot(u);

    Vpred = A.dot(Vfilt).dot(A.T) + Q; # Vpred = Cov[X(t+1) | t]
    J = Vfilt.dot(A.T).dot(pinv(Vpred)); # smoother gain matrix
    # if(cond(Vpred) > 1e+6)
    #     fprintf('ill-conditioned Vpred!\n');
    #     Vpred
    #     Q
    #     A
    #     Vfilt
    # end
    #J = (  (Vpred')\ (A*Vfilt') )'; # smoother gain matrix - changed by PS
    xsmooth = xfilt + J.dot((xsmooth_future - xpred));
    Vsmooth = Vfilt + J.dot((Vsmooth_future - Vpred)).dot(J.T);
    VVsmooth_future = VVfilt_future + (Vsmooth_future - Vfilt_future).dot(pinv(Vfilt_future)).dot(VVfilt_future);
    #VVsmooth_future = VVfilt_future + (VVfilt_future'*((Vfilt_future')\(Vsmooth_future - Vfilt_future)') )';
    return (xsmooth, Vsmooth, VVsmooth_future)


def kalman_smoother(y, A, C, Q, R, init_x, init_V):
    # Kalman/RTS smoother.
    # [xsmooth, Vsmooth, VVsmooth, loglik] = kalman_smoother(y, A, C, Q, R, init_x, init_V)
    #
    # The inputs are the same as for kalman_filter.
    # The outputs are almost the same, except we condition on y(:, 1:T) (and u(:, 1:T) if specified),
    # instead of on y(:, 1:t).

    y = np.asmatrix(y)

    (os, T) = y.shape;
    ss = A.shape[0];

    # set default params
    model = np.ones(T);
    u = np.array([]);
    B = np.array([]);

    xsmooth = np.zeros((ss, T));
    Vsmooth = np.zeros((ss, ss, T));
    VVsmooth = np.zeros((ss, ss, T));

    # Forward pass
    (xfilt, Vfilt, VVfilt, loglik) = kalman_filter(y, A, C, Q, R, init_x, init_V, {'model': model, 'u': u, 'B': B});

    # Backward pass
    xsmooth[:,T-1] = xfilt[:,T-1];
    Vsmooth[:,:,T-1] = Vfilt[:,:,T-1];
    #VVsmooth(:,:,T) = VVfilt(:,:,T);

    for t in range(T-2,-1,-1):
      if B.size == 0:
        (xsmooth[:,t], Vsmooth[:,:,t], VVsmooth[:,:,t+1]) = \
    	smooth_update(xsmooth[:,t+1], Vsmooth[:,:,t+1], xfilt[:,t], Vfilt[:,:,t], \
    		      Vfilt[:,:,t+1], VVfilt[:,:,t+1], A, Q, B, u);
      else:
        (xsmooth[:,t], Vsmooth[:,:,t], VVsmooth[:,:,t+1]) = \
    	smooth_update(xsmooth[:,t+1], Vsmooth[:,:,t+1], xfilt[:,t], Vfilt[:,:,t], \
    		      Vfilt[:,:,t+1], VVfilt[:,:,t+1], A, Q, B, u[:,t+1]);

    VVsmooth[:,:,1] = np.zeros((ss,ss));
    return (xsmooth, Vsmooth, VVsmooth, loglik)

def kalman_update(A, C, Q, R, y, x, V, varargin={}):
    # KALMAN_UPDATE Do a one step update of the Kalman filter
    # [xnew, Vnew, loglik] = kalman_update(A, C, Q, R, y, x, V, ...)
    #
    # INPUTS:
    # A - the system matrix
    # C - the observation matrix
    # Q - the system covariance
    # R - the observation covariance
    # y(:)   - the observation at time t
    # x(:) - E[X | y(:, 1:t-1)] prior mean
    # V(:,:) - Cov[X | y(:, 1:t-1)] prior covariance
    #
    # OPTIONAL INPUTS (string/value pairs [default in brackets])
    # 'initial' - 1 means x and V are taken as initial conditions (so A and Q are ignored) [0]
    # 'u'     - u(:) the control signal at time t [ [] ]
    # 'B'     - the input regression matrix
    #
    # OUTPUTS (where X is the hidden state being estimated)
    #  xnew(:) =   E[ X | y(:, 1:t) ]
    #  Vnew(:,:) = Var[ X(t) | y(:, 1:t) ]
    #  VVnew(:,:) = Cov[ X(t), X(t-1) | y(:, 1:t) ]
    #  loglik = log P(y(:,t) | y(:,1:t-1)) log-likelihood of innovatio

    # set default params
    assert type(y) == np.matrixlib.defmatrix.matrix
    u = np.array([]);
    B = np.array([]);
    initial = 0;
    x = x.reshape(x.size, 1)

    if varargin:
      u = varargin.get('u', u)
      B = varargin.get('B', B)
      initial = varargin['initial']

    #  xpred(:) = E[X_t+1 | y(:, 1:t)]
    #  Vpred(:,:) = Cov[X_t+1 | y(:, 1:t)]

    if initial != 0:
      if u.size == 0:
        xpred = x;
      else:
        xpred = x + B.dot(u);
      Vpred = V;
    else:
      if u.size == 0:
        xpred = A.dot(x);
      else:
        xpred = A.dot(x) + B.dot(u);
      Vpred = A.dot(V).dot(A.T) + Q;

    e = y - C.dot(xpred); # error (innovation)
    ss = A.shape[0];
    S = C.dot(Vpred).dot(C.T) + R;
    if S.shape == ():
      S = np.matrix(S)
    Sinv = pinv(S);
    ss = V.shape[0];
    assert e.shape[0] == S.shape[0]
    loglik = gaussian_prob(e, np.zeros(e.shape[0]), S, 1);
    Vc = Vpred.dot(C.T)
    if len(Vc.shape) == 1:
      Vc = Vc.reshape(Vc.size, 1)
    K = Vc.dot(Sinv); # Kalman gain matrix
    #K = (  (S')\C * Vpred' )'; # Kalman gain matrix - PS
    # If there is no observation vector, set K = zeros(ss).
    xnew = xpred + K.dot(e);
    #print(K.shape, C.shape)
    Vnew = (np.eye(ss) - K.dot(C)).dot(Vpred);
    VVnew = (np.eye(ss) - K.dot(C)).dot(A).dot(V);
    return (xnew.ravel(), Vnew, loglik, VVnew)

def kalman_filter(y, A, C, Q, R, init_x, init_V, varargin={}):
    # Kalman filter.
    # [x, V, VV, loglik] = kalman_filter(y, A, C, Q, R, init_x, init_V, \)
    #
    # INPUTS:
    # y(:,t)   - the observation at time t
    # A - the system matrix
    # C - the observation matrix
    # Q - the system covariance
    # R - the observation covariance
    # init_x - the initial state (column) vector
    # init_V - the initial state covariance
    #
    # OPTIONAL INPUTS (string/value pairs [default in brackets])
    # 'model' - model(t)=m means use params from model m at time t [ones(1,T) ]
    #     In this case, all the above matrices take an additional final dimension,
    #     i.e., A(:,:,m), C(:,:,m), Q(:,:,m), R(:,:,m).
    #     However, init_x and init_V are independent of model(1).
    # 'u'     - u(:,t) the control signal at time t [ [] ]
    # 'B'     - B(:,:,m) the input regression matrix for model m
    #
    # OUTPUTS (where X is the hidden state being estimated)
    # x(:,t) = E[X(:,t) | y(:,1:t)]
    # V(:,:,t) = Cov[X(:,t) | y(:,1:t)]
    # VV(:,:,t) = Cov[X(:,t), X(:,t-1) | y(:,1:t)] t >= 2
    # loglik = sum{t=1}^T log P(y(:,t))
    #
    # If an input signal is specified, we also condition on it:
    # e.g., x(:,t) = E[X(:,t) | y(:,1:t), u(:, 1:t)]
    # If a model sequence is specified, we also condition on it:
    # e.g., x(:,t) = E[X(:,t) | y(:,1:t), u(:, 1:t), m(1:t)]

    assert type(y) == np.matrixlib.defmatrix.matrix
    (os, T) = y.shape;
    ss = A.shape[0]; # size of state space

    # set default params
    model = np.ones(T);
    u = np.array([]);
    B = np.array([]);
    ndx = np.array([]);

    if varargin:
      model = varargin['model']
      u = varargin['u']
      B = varargin['B']
      ndx = varargin.get('ndx', ndx)

    x = np.zeros((ss, T));
    V = np.zeros((ss, ss, T));
    VV = np.zeros((ss, ss, T));

    loglik = 0;
    for t in range(0,T):
      if t==0:
        #prevx = init_x(:,m);
        #prevV = init_V(:,:,m);
        prevx = init_x;
        prevV = init_V;
        initial = 1;
      else:
        prevx = x[:,t-1];
        prevV = V[:,:,t-1];
        initial = 0;

      if u.size == 0:
        assert type(y) == np.matrixlib.defmatrix.matrix
        (xr, Vr, LL, VVr) = kalman_update(A, C, Q, R, y[:,t], prevx, prevV, {'initial': initial});
        x[:,t] = xr
        V[:,:,t] = Vr
        VV[:,:,t] = VVr
      else:
        if ndx.size == 0:
          (x[:,t], V[:,:,t], LL, VV[:,:,t]) = kalman_update(A, C, Q, R, y[:,t], prevx, prevV, {'initial': initial, 'u': u[:,t], 'B': B});
        else:
          i = ndx[t];
          # copy over all elements; only some will get updated
          x[:,t] = prevx;
          prevP = np.inv(prevV);
          prevPsmall = prevP[i,i];
          prevVsmall = np.inv(prevPsmall);
          (x[i,t], smallV, LL, VV[i,i,t]) = kalman_update(A[i,i], C[:,i], Q[i,i], R, y[:,t], prevx[i], prevVsmall, {'initial': initial, 'u': u[:,t], 'B': B[i,:]});
          smallP = np.inv(smallV);
          prevP[i,i] = smallP;
          V[:,:,t] = np.inv(prevP);

      loglik = loglik + LL[0,0];

    return (x, V, VV, loglik)


def learn_kalman(data, A, C, Q, R, initx, initV, max_iter, diagQ, diagR, ARmode, constr_fun, p1, p2, p3):
    # LEARN_KALMAN Find the ML parameters of a stochastic Linear Dynamical System using EM.
    #
    # [A, C, Q, R, INITX, INITV, LL] = LEARN_KALMAN(DATA, A0, C0, Q0, R0, INITX0, INITV0) fits
    # the parameters which are defined as follows
    #   x(t+1) = A*x(t) + w(t),  w ~ N(0, Q),  x(0) ~ N(init_x, init_V)
    #   y(t)   = C*x(t) + v(t),  v ~ N(0, R)
    # A0 is the initial value, A is the final value, etc.
    # DATA(:,t,l) is the observation vector at time t for sequence l. If the sequences are of
    # different lengths, you can pass in a cell array, so DATA{l} is an O*T matrix.
    # LL is the "learning curve": a vector of the log lik. values at each iteration.
    # LL might go positive, since prob. densities can exceed 1, although this probably
    # indicates that something has gone wrong e.g., a variance has collapsed to 0.
    #
    # There are several optional arguments, that should be passed in the following order.
    # LEARN_KALMAN(DATA, A0, C0, Q0, R0, INITX0, INITV0, MAX_ITER, DIAGQ, DIAGR, ARmode)
    # MAX_ITER specifies the maximum number of EM iterations (default 10).
    # DIAGQ=1 specifies that the Q matrix should be diagonal. (Default 0).
    # DIAGR=1 specifies that the R matrix should also be diagonal. (Default 0).
    # ARMODE=1 specifies that C=I, R=0. i.e., a Gauss-Markov process. (Default 0).
    # This problem has a global MLE. Hence the initial parameter values are not important.
    #
    # LEARN_KALMAN(DATA, A0, C0, Q0, R0, INITX0, INITV0, MAX_ITER, DIAGQ, DIAGR, F, P1, P2, ...)
    # calls [A,C,Q,R,initx,initV] = f(A,C,Q,R,initx,initV,P1,P2,...) after every M step. f can be
    # used to enforce any constraints on the params.
    #
    # For details, see
    # - Ghahramani and Hinton, "Parameter Estimation for LDS", U. Toronto tech. report, 1996
    # - Digalakis, Rohlicek and Ostendorf, "ML Estimation of a stochastic linear system with the EM
    #      algorithm and its application to speech recognition",
    #       IEEE Trans. Speech and Audio Proc., 1(4):431--442, 1993.


    #    learn_kalman(data, A, C, Q, R, initx, initV, max_iter, diagQ, diagR, ARmode, constr_fun, varargin)
    verbose = 1;
    thresh = 1e-4;
    N = 1;

    ss = A.shape[0];
    os = C.shape[0];
    data = np.asmatrix(data)
    assert type(data) == np.matrixlib.defmatrix.matrix

    alpha = np.zeros((os, os));
    Tsum = 0;
    for ex in range(N):
      #y = data(:,:,ex);
      y = data;
      #T = length(y);
      T = y.shape[1];
      Tsum = Tsum + T;
      alpha_temp = np.zeros((os, os));
      for t in range(T):
        alpha_temp = alpha_temp + y[:,t].dot(y[:,t].T);

      alpha = alpha + alpha_temp;


    previous_loglik = -1e10;
    loglik = 0;
    converged = 0;
    num_iter = 1;
    LL = np.array([]);

    while not converged and num_iter <= max_iter:

      ### E step

      delta = np.zeros((os, ss));
      gamma = np.zeros((ss, ss));
      gamma1 = np.zeros((ss, ss));
      gamma2 = np.zeros((ss, ss));
      beta = np.zeros((ss, ss));
      P1sum = np.zeros((ss, ss));
      x1sum = np.zeros(ss);
      loglik = 0;

      for ex in range(N):
        y = data;
        T = y.shape[0];
        assert type(y) == np.matrixlib.defmatrix.matrix
        (beta_t, gamma_t, delta_t, gamma1_t, gamma2_t, x1, V1, loglik_t) = Estep(y, A, C, Q, R, initx, initV, ARmode);
        beta = beta + beta_t;
        gamma = gamma + gamma_t;
        delta = delta + delta_t;
        gamma1 = gamma1 + gamma1_t;
        gamma2 = gamma2 + gamma2_t;
        P1sum = P1sum + V1 + x1.dot(x1.T);
        x1sum = x1sum + x1;
        #fprintf(1, 'example #d, ll/T #5.3f\n', ex, loglik_t/T);
        loglik = loglik + loglik_t;

      LL = np.hstack((LL, loglik));
      if verbose:
        print('iteration {}, loglik = {}'.format(num_iter, loglik))

      #fprintf(1, 'iteration #d, loglik/NT = #f\n', num_iter, loglik/Tsum);
      num_iter =  num_iter + 1;

      ### M step

      # Tsum =  N*T
      # Tsum1 = N*(T-1);
      Tsum1 = Tsum - N;
      A = beta.dot(pinv(gamma1));
    #  A = (gamma1' \ beta')'; # PS
      Q = (gamma2 - A.dot(beta.T)) / Tsum1;
      if diagQ:
        Q = np.diag(np.diag(Q));

      if ~ARmode:
        C = delta * pinv(gamma);
    #    C = (gamma' \ delta')'; # PS
        R = (alpha - C.dot(delta.T)) / Tsum;
        if diagR:
          R = np.diag(np.diag(R));

      initx = x1sum[:,0] / N;
      initV = P1sum/N - initx.dot(initx.T);

      if constr_fun != []:
        (A,C,Q,R,initx,initV) = eval('constr_fun(A, C, Q, R, initx, initV, p1, p2, p3)');

      converged, _ = em_converged(loglik, previous_loglik, thresh);
      previous_loglik = loglik;

    return (A, C, Q, R, initx, initV, LL)

#########

def Estep(y, A, C, Q, R, initx, initV, ARmode):
    #
    # Compute the (expected) sufficient statistics for a single Kalman filter sequence.
    #

    assert type(y) == np.matrixlib.defmatrix.matrix
    (os, T) = y.shape;
    ss = A.shape[0];

    if ARmode:
      xsmooth = y.copy();
      Vsmooth = np.zeros((ss, ss, T)); # no uncertainty about the hidden states
      VVsmooth = np.zeros((ss, ss, T));
      loglik = 0;
    else:
      assert type(y) == np.matrixlib.defmatrix.matrix
      (xsmooth, Vsmooth, VVsmooth, loglik) = kalman_smoother(y, A, C, Q, R, initx, initV);

    delta = np.zeros((os, ss));
    gamma = np.zeros((ss, ss));
    beta = np.zeros((ss, ss));
    xsmooth = np.asmatrix(xsmooth)
    for t in range(T):
      delta = delta + y[:,t].dot(xsmooth[:,t].T);
      gamma = gamma + xsmooth[:,t].dot(xsmooth[:,t].T) + Vsmooth[:,:,t];
      if t > 0:
        beta = beta + xsmooth[:,t].dot(xsmooth[:,t-1].T) + VVsmooth[:,:,t];

    gamma1 = gamma - xsmooth[:,T-1].dot(xsmooth[:,T-1].T) - Vsmooth[:,:,T-1];
    gamma2 = gamma - xsmooth[:,0].dot(xsmooth[:,0].T) - Vsmooth[:,:,0];

    x1 = xsmooth[:,0];
    V1 = Vsmooth[:,:,0];

    return (beta, gamma, delta, gamma1, gamma2, x1, V1, loglik)

def gaussian_prob(x, m, C, use_log=0):
    # GAUSSIAN_PROB Evaluate a multivariate Gaussian density.
    # p = gaussian_prob(X, m, C)
    # p(i) = N(X(:,i), m, C) where C = covariance matrix and each COLUMN of x is a datavector

    # p = gaussian_prob(X, m, C, 1) returns log N(X(:,i), m, C) (to prevents underflow).
    #
    # If X has size dxN, then p has size Nx1, where N = number of examples

    if m.size == 1: # scalar
      x = x[:].T;

    (d, N) = x.shape;
    #assert(length(m)==d); # slow
    m = m.reshape(m.size, 1)
    M = m.dot(np.ones((1,N))); # replicate the mean across columns
    C = np.matrix(C)
    denom = (2*np.pi)**(d/2)*np.sqrt(np.abs(det(C)));
    mahal = np.sum(np.multiply(((x-M).T.dot(pinv(C))),(x-M).T),axis=1);   # Chris Bregler's trick
    #mahal = sum(( C'\(x-M) )'.*(x-M)',2);   # Chris Bregler's trick - PS

    if np.any(mahal<0):
      print('mahal < 0 => C is not psd')

    if use_log == 1:
      p = -0.5*mahal - np.log(denom);
    else:
      p = np.exp(-0.5*mahal) / (denom+4e-16);

    return p

def em_converged(loglik, previous_loglik, threshold=1e-4, check_increased=1):
    # EM_CONVERGED Has EM converged?
    # [converged, decrease] = em_converged(loglik, previous_loglik, threshold)
    #
    # We have converged if the slope of the log-likelihood function falls below 'threshold',
    # i.e., |f(t) - f(t-1)| / avg < threshold,
    # where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log lik at iteration t.
    # 'threshold' defaults to 1e-4.
    #
    # This stopping criterion is from Numerical Recipes in C p423
    #
    # If we are doing MAP estimation (using priors), the likelihood can decrase,
    # even though the mode of the posterior is increasing.

    converged = 0;
    decrease = 0;

    if check_increased:
      if loglik - previous_loglik < -1e-3: # allow for a little imprecision
        print('******likelihood decreased from {} to {}!'.format(previous_loglik, loglik));
        decrease = 1;
        converged = 0;
        return (converged, decrease);

    delta_loglik = np.abs(loglik - previous_loglik);
    avg_loglik = (np.abs(loglik) + np.abs(previous_loglik) + 4e-16)/2;
    if (delta_loglik / avg_loglik) < threshold:
       converged = 1;
    return (converged, decrease)

def kalmanLearningConstraint(A, C, Q, R, initx, initV, A0, C0, stage):
    # Resttrictions on the Kalman filter model, which cannot be modified during
    # learning

    for i in range(Q.shape[0]):
        Q[i,i] = max(Q[i,i],0);

    for i in range(R.shape[0]):
        R[i,i] = max(R[i,i],0);

    for i in range(initV.shape[0]):
        #print("initV",initV)
        initV[i,i] = max(initV[i,i],0);

    if stage == 3:
        A = A0;
        Q[2-1,2-1] = 0; Q[3-1,3-1] = 0; Q[5-1,5-1] = 0; Q[7-1,7-1] = 0;
        print(initV);
        #     initx(1:3) = sum(initx(1:3))/3;
        #     initx(4:5)  = sum(initx(4:5))/2;
        #     initx(6:7)  = sum(initx(6:7))/;
        
        #iV = np.diag(initV);
        initV[[2-1, 3-1, 5-1, 7-1]] = 0; # Variance of lags
        #initV = np.diag(iV);

        #Cf = [1, 0; -a1,-b3; -a2,0; -c*a3/2,0; -c*a3/2,0; -a3/2, 0; -a3/2,0];
        # C(1,4) = C(1,5) are the only entries that may be altered (learned). Changing other
        # fields would modify the intercept mu_f, which we can not do a posteriori
        C45 =  0.5*(C[1-1,4-1] + C[1-1,5-1]);
        C = np.matrix(C0)

     #   if abs(C45) < abs(3.75*C(1,6)) || abs(C45) > abs(4.25*C(1,6))
     #       C45 = 4.0*C(1,6); #Constrain c
     #   end
        C[0,3:5] = C45;

        # Correct grates and z such that nrr lies in the reasonable range
        # at the beginning
        nrrmin = 1;
        nrrmax = 5;
        nrrmiddle = 3;
        c = C45/C[1-1,6-1];
        nrr0 = c*initx[4-1] + initx[6-1];

        if nrr0 > nrrmax or nrr0 < nrrmin:
            # a) modify z only
            initx[6-1] = nrrmiddle - c*initx[4-1];
            initx[7-1] = initx[6-1];
            #
            # b) scale both z and g_rate
#             initx(4) = initx(4)*nrrmiddle/nrr0;
#             initx(6) = initx(6)*nrrmiddle/nrr0;
#             initx(5) = initx(4);
#             initx(7) = initx(6);
#                     initV(4,4) = initV(4,4)*(nrrmiddle/nrr0)^2;
#             initV(6,6) = initV(6,6)*(nrrmiddle/nrr0)^2;

    elif stage == 2:
        
        A = A0;
        Q[2-1,2-1] = 0; Q[3-1,3-1] = 0;
        Q[5-1,5-1] = 0;
        #iV = np.diag(initV);
        #print ("initV", (len(initV)));
        #print("IV",iV)
        initV[2-1:3] = 0; initV[5-1] = 0;
        #initV = np.diag(iV);

        C14 = C[1-1,4-1];
        C = C0;
        C[1-1,4-1] = C14;
    elif stage == 1:
        A = A0;
        C = C0;
        Q[2-1,2-1] = 0; Q[3-1,3-1] = 0;
        Q[4-1,4-1] = 0;
        initV[2-1,2-1] = 0; initV[3-1,3-1] = 0; initV[4-1,4-1] = 0;

    return (A,C,Q,R,initx,initV)
