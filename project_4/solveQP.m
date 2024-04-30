function [U, lam] = solveQP(H, q, A, b, lam0)
  % Implementation of Dual Projected Gradient to solve QP problem from HW4
  % Compute useful A_invH term (compute once to save cost)
  A_invH = A / H;

  % Compute dual gradient projection variables
  Hd = A_invH * A';
  qd = A_invH * q + b;

  % Prepare for iterations of optimization
  Nit = 50; % Terminate after 50 iterations
  lam = lam0; % Inialize vector of dual variables
  L = norm(Hd); % Compute L for dual gradient projection
  k = 1; % Set iteration count
  df = Hd * lam + qd;

  % Run iterations of finding new lamda values
  while k <= Nit
    % Compute lam k+1
    lam = max(lam - 1 / L * df, 0);
    % Updated df and k
    df = Hd * lam + qd;
    k = k + 1;
  end

  % Provide solution after 50 iterations
  U = -inv(H) * (A' * lam + q);
end
