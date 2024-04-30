% coupled gimbal  dynamics
clc; clear; close all;

%% Constants
p.I_I_yy = 2.8813; % pg.23
p.sigma_0IO = 0.1; % friction between inner and outer gimbals(lookup table)
noise_proc = 0.000001; % noise in z
noise_meas = 0.000001; % noise in y

%% Set Simulation Parameters
nx_coupled = 6;
nu_coupled = 2;
nd_coupled = 8;

Nsim = 100;
t = 0;
Ts = 0.1; % sec
h = Ts / 10; % hold each command for 100th of a second

%% Define Continuous System (Linearization)
syms eta eps Omega_OBz Omega_IOy z_OB z_IO
x_sym = [eta; eps; Omega_OBz; Omega_IOy; z_OB; z_IO];
syms T_M0 T_MI
u_sym = [T_M0; T_MI];
syms theta phi psi_dot theta_dot phi_dot psi_ddot theta_ddot phi_ddot
w_sym = [theta; phi; psi_dot; theta_dot; phi_dot; psi_ddot; ...
           theta_ddot; phi_ddot];

% Set up symbolic problem
f = dynamics_w_dist(0, x_sym, u_sym, w_sym);
H = output(x_sym, u_sym, w_sym);
A = jacobian(f, x_sym);
B = jacobian(f, u_sym);
Bw = jacobian(f, w_sym);
C = jacobian(H, x_sym);
Cw = jacobian(H, w_sym);

% Solver linearization point ALL ZEROS
x0 = ones(nx_coupled, 1) * 1e-5;
u0 = ones(nu_coupled, 1) * 1e-5;
w0 = ones(nd_coupled, 1) * 1e-5;
Ac = double(subs(A, [x_sym; u_sym; w_sym], [x0; u0; w0]));
Bc = double(subs(B, [x_sym; u_sym; w_sym], [x0; u0; w0]));
Bwc = double(subs(Bw, [x_sym; u_sym; w_sym], [x0; u0; w0]));
Cc = double(subs(C, [x_sym; u_sym; w_sym], [x0; u0; w0]));
Cwc = double(subs(Cw, [x_sym; u_sym; w_sym], [x0; u0; w0]));
Dc = zeros(size(Cc, 1), size(Bc, 2));

% Check Observability and Controllability of matrix
fprintf("Linarized Continuous Model has controllability rank %.0f\n", ...
  rank(ctrb(Ac, Bc)))
fprintf("Linarized Continuous Model has observability rank %.0f\n", ...
  rank(obsv(Ac, Cc)))

% Define discrete system
% Need to figure out how to add the disturbances probably will be fixed by
% modifying them as states or inputs
sysd = c2d(ss(Ac, Bc, Cc, Dc), Ts);
Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;
Dd = sysd.D;

% Check Observability and Controllability of matrix
fprintf("Linearized Discrete Model has controllability rank %.0f\n", ...
  rank(ctrb(Ad, Bd)))
fprintf("Linearized Discrete Model has observability rank %.0f\n", ...
  rank(obsv(Ad, Cd)))

t = 0;
% Initalize continous dynamic states
x0_G = zeros(nx_coupled, 1);
u0_G = zeros(nu_coupled, 1);

% High frequency test case
wParams.at = 0.03;
wParams.ft = 1.3;
wParams.aph = 0.11;
wParams.fph = 1.2;
wParams.aps = 0.013;
wParams.fps = 1.15;
w0_G = disturbance(0, wParams);

N = 20; % MPC horizon
ref = [1; 0.25; 0; 0]; %zeros(4,1); % Set reference

% ---------------- MPC Formulation 4 --------------------
% Obtain the QP matrix for formulation 4
[H, L, G, W, T, IMPC] = formulate4(Ad, Bd, Cd, N);
sW = size(W);
% Initalize MPC controller
lam0 = ones(sW(1), 1);

y0_MPC = output(x0_G, u0_G, w0_G); % Sensor Measurements
x0_MPC = [zeros(nx_coupled, 1); y0_MPC - ref; x0_G; u0_G]; % Extended States
% -------------------------------------------------

% Initalize arrays to store results
data = [];
data.x(1, :) = x0_G;
data.u(1, :) = u0_G;
data.w(1, :) = w0_G;
data.t(1, :) = 0;

% r = zeros(nx_coupled,1);
for ii = 1:1:Nsim %simulate over XXXXX seconds

  % Solve Quadratic programing using function from homework 4
  [U, lam] = solveQP(H, L * x0_MPC, G, W + T * x0_MPC, lam0);
  lam0 = lam; % For warm start next time

  % Convert from increment du to full u
  du = IMPC * U;
  u_G = du + u0_G;

  % perform integration over the [t, t+Ts]
  interval = t + h:h:t + Ts;

  options = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);
  u = u_G; % control is fixed over interval
  % w = w_G; % disturbance static over interval

  % Simulate Dynamics
  [t_array, x_G] = ode45(@(t, x) dynamics(t, x, u, wParams), ...
    interval, x0_G);

  % [t_arrayl, x_G_and_w_G] = ode45(@(t, x_G_and_w) coupled_gimbal_xw(t, x_G_and_w, u,wParams),...
  %                         interval, x0_G_and_w0_G);

  % Store Iteration Data
  t = t + Ts;
  w_G = disturbance(t, wParams);

  data.x(ii + 1, :) = x_G(end, :);
  data.u(ii + 1, :) = u_G';
  data.w(ii + 1, :) = w_G';
  data.t(ii + 1, :) = t;

  % Update MPC States
  x = x_G(end, :)'; % Need to switch to observation eventually
  dx = x - x0_G; % Need to switch to observation eventually
  y_MPC = output(x, u_G, w_G);
  e = y_MPC - ref;
  x0_MPC = [dx; e; x; u];

  % Update States and Controls
  x0_G = x_G(end, :)';
  u0_G = u_G;

  % Print updates on simulation
  if mod(ii, 20) == 0
    pdone = 100 * ii / Nsim;
    fprintf("%.0f percent done\n", pdone)
  end

end

% Plot Simulation Results
% close all
plot_sys(data);
% analysis_MPC(data);

%% Dynamics Functions
% Nonlinear Dynamics Equations (Used for linearization)
function state_dot = dynamics_w_dist(t, x_G, u, w)

  % unpack variables
  % states of coupled model, constraints enforced on these
  state_coupled = x_G;
  x_G1 = state_coupled(1); % eta
  x_G2 = state_coupled(2); % eps
  x_G3 = state_coupled(3); % Omega_OBz
  x_G4 = state_coupled(4); % Omega_IOy
  x_G5 = state_coupled(5); % z_OB
  x_G6 = state_coupled(6); % z_IO

  eta = state_coupled(1);
  eps = state_coupled(2);
  Omega_OBz = state_coupled(3);
  Omega_IOy = state_coupled(4);
  z_OB = state_coupled(5);
  z_IO = state_coupled(6);

  % control of coupled model, constraints enforced on these
  u_G1 = u(1);
  u_G2 = u(2); % motor torque trasmitted to inner gimbal

  % disburbances of coupled model, constraints enforced on these
  dist_coupled = w;
  w_G1 = dist_coupled(1);
  w_G2 = dist_coupled(2);
  w_G3 = dist_coupled(3);
  w_G4 = dist_coupled(4);
  w_G5 = dist_coupled(5);
  w_G6 = dist_coupled(6);
  w_G7 = dist_coupled(7);
  w_G8 = dist_coupled(8);
  theta = dist_coupled(1);
  phi = dist_coupled(2);
  psi_dot = dist_coupled(3);
  theta_dot = dist_coupled(4);
  phi_dot = dist_coupled(5);
  psi_ddot = dist_coupled(6);
  theta_ddot = dist_coupled(7);
  phi_ddot = dist_coupled(8);

  % eqns 2.53, 2.54 pg. 35
  eta_dot = Omega_OBz;
  eps_dot = Omega_IOy;

  %%% pg. 38 eqns, LHS
  s_g1 = 1 + 1.441 * cos(x_G2)^2 - 1.451 * sin(x_G2)^2;
  s_g2 = 0.0364 * cos(x_G2) + 0.0052 * sin(x_G2);
  s_g3 = -0.02345 * sin(x_G2) + 0.1274 * cos(x_G2);

  S_G = [1, 0, 0, 0, 0, 0;
         0, 1, 0, 0, 0, 0;
         0, 0, s_g1, s_g2, 0, 0;
         0, 0, s_g3, 1, 0, 0;
         0, 0, 0, 0, 1, 0;
         0, 0, 0, 0, 0, 1];

  % angular velocity vector of the outer gimbal
  Omega_O_ex = phi_dot * cos(phi) + theta_dot * cos(phi) * sin(eta) + psi_dot * (cos(theta) * sin(eta) * sin(phi) - cos(eta) * sin(theta)); % eqn 2.13
  Omega_O_ey = -phi_dot * sin(phi) + theta_dot * cos(phi) * cos(eta) + psi_dot * (sin(eta) * sin(theta) + cos(eta) * cos(theta) * sin(phi)); % eqn 2.14

  Omega_dist_Oz = -theta_dot * sin(phi) + psi_dot * cos(phi) * cos(theta); % disturbance velocity effect about the z axis of the outer gimbal

  % define angular accels of outer gimbal about its body frame axes
  % state_I = [eps; Omega_IOy; z_IO];
  % control_I = T_mI;
  alphas = angular_accelerations(x_G, u, w);
  alpha_O_ex = alphas(1);
  alpha_O_ey = alphas(2);
  alpha_dist_Ix = alphas(3);
  alpha_dist_Iy = alphas(4);
  alpha_dist_Iz = alphas(5);
  alpha_dist_Oz = alphas(6);

  % define coupled rates
  T_fr_IO = 90 * x_G6; % friction torque bw inner and outer gimbals; pg. 37
  T_fr_BO = 80 * x_G5; % pg. 36, friction torque bw gimbal base and outer gimbal

  % static mass unbalance
  D_O_staticUnbz = -0.043 * (-cos(x_G1) * sin(w_G1) + sin(x_G1) * sin(w_G2) * cos(w_G1)) + ...
    164.8 * (sin(x_G1) * sin(w_G1) + cos(x_G1) * sin(w_G2) * cos(w_G1));

  % define inner gimbal rates: eqn 2.53, 2.55, 2.57
  % define Omega_IOy_dot
  % eqn 2.17-2.18, inner gimbal angular velocities, wrt to Earth frame
  Omega_I_ez = phi_dot * cos(eps) * cos(eta) - ...
    psi_dot * (sin(eps) * cos(phi) - cos(eps) * sin(eta) * sin(phi)) + ...
    theta_dot * (sin(eps) * sin(phi) + cos(eps) * sin(eta) * cos(phi)) - eta_dot * sin(eps);
  Omega_dist_Iy = -phi_dot * sin(eta) + theta_dot * cos(eta) * cos(phi) + psi_dot * cos(eta) * sin(phi); % disturbance velocity about y axis of inner gimbal
  Omega_I_ex = eps_dot + Omega_dist_Iy;

  Dhat_O_dynUnbz = -4.76 * alpha_O_ex + 0.0006 * alpha_O_ey + ...
    alpha_dist_Ix * (0.052 * cos(x_G2) + 14.51 * sin(x_G2)) + ...
    alpha_dist_Iy * (0.364 * cos(x_G2) + 0.052 * sin(x_G2)) + ...
    alpha_dist_Iz * (14.41 * cos(x_G2) + 0.052 * sin(x_G2)) + ...
    0.07 * (Omega_O_ex)^2 - (Omega_O_ey^2) * (0.07 * cos(x_G2) - 0.36 * sin(x_G2)) + ...
    0.3606 * Omega_O_ex * (x_G3 + Omega_dist_Oz) - ...
    Omega_O_ey * Omega_O_ex * (-9.235 + 14.41 * cos(x_G2) - 0.052 * sin(x_G2)) - ...
    Omega_O_ey * (x_G3 + Omega_dist_Oz) * (-4.76 + 0.052 * cos(x_G2) - 14.41 * sin(x_G2)) - ...
    Omega_I_ex * x_G4 * (14.51 * cos(x_G2) - 0.052 * sin(x_G2)) - ...
    (x_G4 + Omega_dist_Iy) * x_G4 * (0.07 * cos(x_G2) - 0.37 * sin(x_G2)) - ...
    Omega_I_ez * x_G4 * (0.052 * cos(x_G2) - 14.41 * sin(x_G2));

  % static mass imbalance for inner gimbal
  D_I_staticUnby = -2.66 * (cos(x_G2) * cos(w_G1) - sin(x_G2) * sin(w_G1)) - ...
    6.44 * (cos(x_G2) * sin(w_G1) + cos(w_G1) * sin(x_G2));

  temp3 = Omega_I_ez * (14.51 * Omega_I_ex + 0.067 * (x_G4 + Omega_dist_Iy) + 0.052 * Omega_I_ez);
  temp4 = Omega_I_ex * (0.052 * Omega_I_ex + 0.364 * (x_G4 + Omega_dist_Iy) + 14.41 * Omega_I_ez);
  Dhat_I_dynUnby = 0.067 * alpha_dist_Ix + 0.364 * alpha_dist_Iz + temp3 + temp4;

  T_fr_OI = 90 * x_G6; %friction torque bw inner and outer gimbal

  %%% pg. 39 eqns, RHS
  eqn1 = x_G3;
  eqn2 = x_G4;
  eqn3 = -0.1 * Dhat_O_dynUnbz + 0.1 * u_G1 + 0.1 * T_fr_BO + 0.1 * T_fr_IO + 0.1 * D_O_staticUnbz - alpha_dist_Oz; % eqn 2.55;
  eqn4 = -0.35 * Dhat_I_dynUnby + 0.35 * T_fr_OI + 0.35 * u_G2 + 0.35 * D_I_staticUnby - alpha_dist_Iy; % eqn 2.56
  eqn5 = x_G3 - 4.42 * abs(x_G3) * x_G5; % eqn 2.57
  eqn6 = x_G4 - 33.3 * abs(x_G4) * x_G6; % eqn 2.58
  b = [eqn1; eqn2; eqn3; eqn4; eqn5; eqn6];

  xdot_G = S_G \ b;

  state_dot = [xdot_G; ];
end

% Computes distrubances analytically (use for propagating dynamics)
function state_dot = dynamics(t, x_G, u, distParams)

  % unpack variables
  % states of coupled model, constraints enforced on these
  state_coupled = x_G;
  x_G1 = state_coupled(1); % eta
  x_G2 = state_coupled(2); % eps
  x_G3 = state_coupled(3); % Omega_OBz
  x_G4 = state_coupled(4); % Omega_IOy
  x_G5 = state_coupled(5); % z_OB
  x_G6 = state_coupled(6); % z_IO

  eta = state_coupled(1);
  eps = state_coupled(2);
  Omega_OBz = state_coupled(3);
  Omega_IOy = state_coupled(4);
  z_OB = state_coupled(5);
  z_IO = state_coupled(6);

  % control of coupled model, constraints enforced on these
  u_G = u;
  u_G1 = u_G(1);
  u_G2 = u_G(2); % motor torque trasmitted to inner gimbal

  % disburbances of coupled model, constraints enforced on these
  disturbances = disturbance(t, distParams);
  w_G1 = disturbances(1);
  w_G2 = disturbances(2);
  w_G3 = disturbances(3);
  w_G4 = disturbances(4);
  w_G5 = disturbances(5);
  w_G6 = disturbances(6);
  w_G7 = disturbances(7);
  w_G8 = disturbances(8);
  theta = disturbances(1);
  phi = disturbances(2);
  psi_dot = disturbances(3);
  theta_dot = disturbances(4);
  phi_dot = disturbances(5);
  psi_ddot = disturbances(6);
  theta_ddot = disturbances(7);
  phi_ddot = disturbances(8);

  % eqns 2.53, 2.54 pg. 35
  eta_dot = Omega_OBz;
  eps_dot = Omega_IOy;

  %%% pg. 38 eqns, LHS
  s_g1 = 1 + 1.441 * cos(x_G2)^2 - 1.451 * sin(x_G2)^2;
  s_g2 = 0.0364 * cos(x_G2) + 0.0052 * sin(x_G2);
  s_g3 = -0.02345 * sin(x_G2) + 0.1274 * cos(x_G2);

  S_G = [1, 0, 0, 0, 0, 0;
         0, 1, 0, 0, 0, 0;
         0, 0, s_g1, s_g2, 0, 0;
         0, 0, s_g3, 1, 0, 0;
         0, 0, 0, 0, 1, 0;
         0, 0, 0, 0, 0, 1];

  % angular velocity vector of the outer gimbal
  Omega_O_ex = phi_dot * cos(phi) + theta_dot * cos(phi) * sin(eta) + psi_dot * (cos(theta) * sin(eta) * sin(phi) - cos(eta) * sin(theta)); % eqn 2.13
  Omega_O_ey = -phi_dot * sin(phi) + theta_dot * cos(phi) * cos(eta) + psi_dot * (sin(eta) * sin(theta) + cos(eta) * cos(theta) * sin(phi)); % eqn 2.14

  Omega_dist_Oz = -theta_dot * sin(phi) + psi_dot * cos(phi) * cos(theta); % disturbance velocity effect about the z axis of the outer gimbal

  % define angular accels of outer gimbal about its body frame axes
  % state_I = [eps; Omega_IOy; z_IO];
  % control_I = T_mI;
  alphas = angular_accelerations(x_G, u, disturbances);
  alpha_O_ex = alphas(1);
  alpha_O_ey = alphas(2);
  alpha_dist_Ix = alphas(3);
  alpha_dist_Iy = alphas(4);
  alpha_dist_Iz = alphas(5);
  alpha_dist_Oz = alphas(6);

  % define coupled rates
  T_fr_IO = 90 * x_G6; % friction torque bw inner and outer gimbals; pg. 37
  T_fr_BO = 80 * x_G5; % pg. 36, friction torque bw gimbal base and outer gimbal

  % static mass unbalance
  D_O_staticUnbz = -0.043 * (-cos(x_G1) * sin(w_G1) + sin(x_G1) * sin(w_G2) * cos(w_G1)) + ...
    164.8 * (sin(x_G1) * sin(w_G1) + cos(x_G1) * sin(w_G2) * cos(w_G1));

  % define inner gimbal rates: eqn 2.53, 2.55, 2.57
  % define Omega_IOy_dot
  % eqn 2.17-2.18, inner gimbal angular velocities, wrt to Earth frame
  Omega_I_ez = phi_dot * cos(eps) * cos(eta) - ...
    psi_dot * (sin(eps) * cos(phi) - cos(eps) * sin(eta) * sin(phi)) + ...
    theta_dot * (sin(eps) * sin(phi) + cos(eps) * sin(eta) * cos(phi)) - eta_dot * sin(eps);
  Omega_dist_Iy = -phi_dot * sin(eta) + theta_dot * cos(eta) * cos(phi) + psi_dot * cos(eta) * sin(phi); % disturbance velocity about y axis of inner gimbal
  Omega_I_ex = eps_dot + Omega_dist_Iy;

  Dhat_O_dynUnbz = -4.76 * alpha_O_ex + 0.0006 * alpha_O_ey + ...
    alpha_dist_Ix * (0.052 * cos(x_G2) + 14.51 * sin(x_G2)) + ...
    alpha_dist_Iy * (0.364 * cos(x_G2) + 0.052 * sin(x_G2)) + ...
    alpha_dist_Iz * (14.41 * cos(x_G2) + 0.052 * sin(x_G2)) + ...
    0.07 * (Omega_O_ex)^2 - (Omega_O_ey^2) * (0.07 * cos(x_G2) - 0.36 * sin(x_G2)) + ...
    0.3606 * Omega_O_ex * (x_G3 + Omega_dist_Oz) - ...
    Omega_O_ey * Omega_O_ex * (-9.235 + 14.41 * cos(x_G2) - 0.052 * sin(x_G2)) - ...
    Omega_O_ey * (x_G3 + Omega_dist_Oz) * (-4.76 + 0.052 * cos(x_G2) - 14.41 * sin(x_G2)) - ...
    Omega_I_ex * x_G4 * (14.51 * cos(x_G2) - 0.052 * sin(x_G2)) - ...
    (x_G4 + Omega_dist_Iy) * x_G4 * (0.07 * cos(x_G2) - 0.37 * sin(x_G2)) - ...
    Omega_I_ez * x_G4 * (0.052 * cos(x_G2) - 14.41 * sin(x_G2));

  % static mass imbalance for inner gimbal
  D_I_staticUnby = -2.66 * (cos(x_G2) * cos(w_G1) - sin(x_G2) * sin(w_G1)) - ...
    6.44 * (cos(x_G2) * sin(w_G1) + cos(w_G1) * sin(x_G2));

  temp3 = Omega_I_ez * (14.51 * Omega_I_ex + 0.067 * (x_G4 + Omega_dist_Iy) + 0.052 * Omega_I_ez);
  temp4 = Omega_I_ex * (0.052 * Omega_I_ex + 0.364 * (x_G4 + Omega_dist_Iy) + 14.41 * Omega_I_ez);
  Dhat_I_dynUnby = 0.067 * alpha_dist_Ix + 0.364 * alpha_dist_Iz + temp3 + temp4;

  T_fr_OI = 90 * x_G6; %friction torque bw inner and outer gimbal

  %%% pg. 39 eqns, RHS
  eqn1 = x_G3;
  eqn2 = x_G4;
  eqn3 = -0.1 * Dhat_O_dynUnbz + 0.1 * u_G1 + 0.1 * T_fr_BO + 0.1 * T_fr_IO + 0.1 * D_O_staticUnbz - alpha_dist_Oz; % eqn 2.55;
  eqn4 = -0.35 * Dhat_I_dynUnby + 0.35 * T_fr_OI + 0.35 * u_G2 + 0.35 * D_I_staticUnby - alpha_dist_Iy; % eqn 2.56
  eqn5 = x_G3 - 4.42 * abs(x_G3) * x_G5; % eqn 2.57
  eqn6 = x_G4 - 33.3 * abs(x_G4) * x_G6; % eqn 2.58
  b = [eqn1; eqn2; eqn3; eqn4; eqn5; eqn6];

  xdot_G = S_G \ b;

  state_dot = [xdot_G; ];
end
