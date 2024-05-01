classdef EKF < handle

  properties
    % prior for the current step
    prior_mean
    prior_cov
    % dynamics and measurement for predict and update
    dynamics
    measure
    % noise covariances
    cov_process
    cov_measure
    % steps that have been run
    step_time
    steps_curr = 1
    % data store to store mean, std and covariance
    store_mean
    store_std
    store_cov
  end

  methods

    function obj = EKF(prior_mean, prior_cov, dynamics, cov_process, measure, cov_measure, step_time)
      obj.prior_mean = prior_mean;
      obj.prior_cov = prior_cov;
      obj.dynamics = dynamics;
      obj.cov_process = cov_process;
      obj.measure = measure;
      obj.cov_measure = cov_measure;
      obj.step_time = step_time;
      obj.store_mean(1, :) = prior_mean;
      obj.store_std(1, :) = sqrt(diag(prior_cov));
      obj.store_cov(1, :, :) = prior_cov;
      obj.steps_curr = obj.steps_curr + 1;
    end

    function filter(obj, u, w, Ak, Hk, measurement)
      % predict
      mk_ = obj.prior_mean + obj.step_time * obj.dynamics(0, obj.prior_mean, u, w); % keep t as 0, unused
      Ck_ = Ak * obj.prior_cov * Ak' + obj.cov_process;

      % check if new measurement exists
      if exist('measurement', 'var')
        % update
        mu = obj.measure(mk_, u, w);
        U = Ck_ * Hk';
        S = Hk * U + obj.cov_measure;
        mk = mk_ + U * (S \ (measurement - mu));
        Ck = Ck_ - U * (S \ U');
      else
        % use predicted values
        mk = mk_;
        Ck = ck_;
      end

      obj.prior_mean = mk;
      obj.prior_cov = Ck;

      obj.store_mean(obj.steps_curr, :) = mk;
      obj.store_std(obj.steps_curr, :) = sqrt(diag(Ck));
      obj.store_cov(obj.steps_curr, :, :) = Ck;
      obj.steps_curr = obj.steps_curr + 1;
    end

  end

end
