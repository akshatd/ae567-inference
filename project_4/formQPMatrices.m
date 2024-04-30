function [H, L, G, W, T, IMPC] = ...
     formQPMatrices(A, B, Q, R, P, xlim, ulim, N)
% Forms matrix for QP problem given a constrained LQ-MPC problem
% Function written for MPC Homework 4, problem 1 part G (Noah Burns)
% Confirmed against solution
% Minimize 1/2*U^T H U + q^{\rm T} U subject to G U <= W + T*x_0
% where q = L*x_0
% The matrix IMPC extracts the first control move, u_MPC=IMPC*U
% N is the horizon, xlim, ulim are limits on x and u

% Obtain dimensions nx and nu
[nx, nu] = size(B);

% Initialize Matrix S and M
S    = [];
M    = [];

% Generate values for S and M
SRow = zeros(nx,N*nu);
for ii = 1:1:N
    M = [M; A^ii];
    SRow = [A^(ii-1)*B SRow(1:nx,1:end-nu)]; % Add new element to Srow
    S = [S; SRow];
end

% Initialize Qbar and Rbar
Qbar = zeros(nx*N,nx*N);
Rbar = zeros(nu*N,nu*N);

% Compute Qbar and Rbar
for ii = 1:1:N
    Qbar(1+nx*(ii-1):nx*ii,1+nx*(ii-1):nx*ii) = Q;
    Rbar(1+nu*(ii-1):nu*ii,1+nu*(ii-1):nu*ii) = R;
end
Qbar(1+nx*(ii-1):nx*ii,1+nx*(ii-1):nx*ii) = P;

% Compute H
H = S'*Qbar*S+Rbar;

% Compute L
L = S'*Qbar*M;

% Compute G
G = [S;-S;eye(N*nu);-eye(N*nu)];

% Compute bounds
Xmax = repmat(xlim.max',N,1);
Xmin = repmat(xlim.min',N,1);
Umax = repmat(ulim.max',N,1);
Umin = repmat(ulim.min',N,1);

% Compute W
W = [Xmax; -Xmin; Umax; -Umin];

% Compute T
T = [-M;M;zeros(N*nu,nx);zeros(N*nu,nx)];

% Compute IMPC
IMPC = [eye(nu,nu), zeros(nu,(N-1)*nu)];
end