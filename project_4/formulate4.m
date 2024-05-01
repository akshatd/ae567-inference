function [H, L, G, W, T, IMPC] = formulate4(A, B, C, N)
  % Computes forumaltion 4 for the constrainted LQ-MPC Problem
  %   Formulation 4 is the incremental (rate) approach
  %   Returns the matrices for the QP problem
  %   Inputs: A - linearized discrete model for the states
  %           B - linearized discrete model for the inputs
  %           C - linearized discrete model for the observations
  %           N - forcast horizon for QP problem
  %   Outputs: matrices defining the QP problem
  %% Define Augmented Problem
  % Define dimensions for problem
  nx = size(A, 1); % Number of states
  nu = size(B, 2); % Number of inputs
  ny = size(C, 1); % Number of measurements
  ne = ny; % Number of error (equivalent to measurments)

  % Format augmented A, B, C, D arrays according to Incremental model (form4)
  % Extend states are [Delta x_k, e_k, x_k-1, u_k-1]
  Aext = [A, zeros(nx, ne), zeros(nx, nx), zeros(nx, nu); ...
            C * A, eye(ne), zeros(ne, nx), zeros(ne, nu); ...
            eye(nx), zeros(nx, ne), eye(nx), zeros(nx, nu); ...
            zeros(nu, nx), zeros(nu, ne), zeros(nu, nx), eye(nu)];

  Bext = [B; C * B; zeros(nx, nu); eye(nu)];

  % Cost Matrix for QP Problem ---------------------------------------------
  % Cost on augmented states
  % Qe   = eye(ne)*10; % Equal penality to all error (Tunable)
  Qe = blkdiag([100], [100], zeros(2, 2));
  % Cost on delta u
  R = eye(nu); % Equal penality to all control increments (Tunable
  R(1, 1) = 0.01;
  R(2, 2) = 0.01;
  % Augmented
  Qext = blkdiag(zeros(nx), Qe, zeros(nx), zeros(nu));
  Rext = R;
  % ---------------------------------------------------------------------

  % Compute terminal penality using smaller system
  A_ang = A(1:2, 1:2);
  B_ang = B(1:2, :);
  C_ang = C(1:2, 1:2);
  % Aterm = [A_ang, zeros(nx - 4, ne - 2); ...
  %            C_ang * A_ang, eye(ne - 2)];
  Aterm = [A_ang, zeros(2, 2); C_ang * A_ang, eye(2)];
  Bterm = [B_ang; C_ang * B_ang];
  % Qterm = blkdiag(zeros(nx - 4), Qe(1:2, 1:2));
  Qterm = blkdiag(zeros(2), Qe(1:2, 1:2));
  % Aterm       = [A,  zeros(nx,ne);...
  %                C*A, eye(ne)];
  % Bterm       = [B; C*B];
  % Qterm       = blkdiag(zeros(nx), Qe);
  [Pdxu, ~, ~] = idare(Aterm, Bterm, Qterm, R);

  Pinf = zeros(nx + ne);
  Pinf(1:2, 1:2) = Pdxu(1:2, 1:2);
  Pinf(1:2, nx + 1:nx + 2) = Pdxu(1:2, 3:4);
  Pinf(nx + 1:nx + 2, 1:2) = Pdxu(3:4, 1:2);
  Pinf(nx + 1:nx + 2, nx + 1:nx + 2) = Pdxu(3:4, 3:4);
  Pext = blkdiag(zeros(nx + ny), zeros(nx), zeros(nu));
  Pext = blkdiag(Pinf, zeros(nx), zeros(nu));

  % Set constraints for QP problem
  ln = 1000;
  xMax = [pi, 4 * pi / 9, 20, 12, 10, 10];
  xMin = [-pi, -pi / 9, -20, -12, -10, -10];
  uMax = [203, 35.3];
  uMin = -uMax;
  xlim.max = [ones(1, nx + ne) * ln, xMax, uMax];
  xlim.min = [ones(1, nx + ne) * -ln, xMin, uMin];
  ulim.max = [ln, ln];
  ulim.min = -ulim.max;

  %% Compute inputs for QP Problem
  [H, L, G, W, T, IMPC] = formQPMatrices(Aext, Bext, Qext, Rext, Pext, ...
    xlim, ulim, N);
end
