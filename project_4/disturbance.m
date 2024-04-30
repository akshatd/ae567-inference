% Compute analytical disturbances
function w = disturbance(t, w_params)
  % Extract parameters
  at = w_params.at;
  aph = w_params.aph;
  aps = w_params.aps;
  ft = w_params.ft;
  fph = w_params.fph;
  fps = w_params.fps;

  % Compute Disturbances
  theta = at * cos(ft * t);
  phi = aph * cos(fph * t);
  psi_dot = aps * fps * -sin(fps * t);
  theta_dot = at * ft * -sin(ft * t);
  phi_dot = aph * fph * -sin(fph * t);
  psi_ddot = aps * fps^2 * -cos(fps * t);
  theta_ddot = at * ft^2 * -cos(ft * t);
  phi_ddot = aph * fph^2 * -cos(fph * t);
  w = [theta, phi, psi_dot, theta_dot, phi_dot, psi_ddot, theta_ddot, ...
         phi_ddot];
end
