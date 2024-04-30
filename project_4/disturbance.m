% Compute analytical disturbances
function w = disturbance(t, distParams)
  % Extract parameters
  at = distParams.at;
  aph = distParams.aph;
  aps = distParams.aps;
  ft = distParams.ft;
  fph = distParams.fph;
  fps = distParams.fps;

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
