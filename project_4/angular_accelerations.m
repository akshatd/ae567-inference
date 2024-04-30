function alphas = angular_accelerations(x, u, w)

  % unpack variables
  eta = x(1);
  eps = x(2);
  Omega_OBz = x(3);
  Omega_IOy = x(4);
  z_OB = x(5);
  z_IO = x(6);

  % control of coupled model, constraints enforced on these
  u_G = u;
  T_mO = u_G(1);
  T_mI = u_G(2); % motor torque trasmitted to inner gimbal

  % disburbances of coupled model, constraints enforced on these
  theta = w(1);
  phi = w(2);
  psi_dot = w(3);
  theta_dot = w(4);
  phi_dot = w(5);
  psi_ddot = w(6);
  theta_ddot = w(7);
  phi_ddot = w(8);

  eta_dot = Omega_OBz;
  eps_dot = Omega_IOy;

  % eqn A1
  alpha_O_ex = phi_ddot * cos(eta) + theta_ddot * cos(phi) * sin(eta) - phi_ddot * (cos(eta) * sin(theta) + cos(theta) * sin(eta) * sin(phi)) + ...
    eta_dot * (phi_dot * sin(eta) - phi_dot * (sin(eta) * sin(theta) + cos(eta) * cos(theta) * sin(phi)) - theta_dot * cos(eta) * cos(phi)) + ...
    phi_dot * theta_dot * ((cos(eta)^2 - cos(phi) * sin(eta)^2) * cos(theta) + (1 + cos(eta)) * sin(eta) * sin(phi) * sin(theta)) * cos(phi) + ...
    phi_dot * (theta_dot * (sin(phi) * cos(eta) + cos(phi) * sin(eta)) * sin(phi) -psi_dot * (cos(phi) * sin(eta) + cos(eta) * sin(phi)) * cos(phi) * cos(theta));

  % eqn A2
  alpha_O_ey = theta_ddot * cos(eta) * cos(phi) - phi_ddot * sin(eta) + phi_ddot * (sin(eta) * sin(theta) + cos(eta) * cos(theta) * sin(phi)) + ...
    eta_dot * (phi_dot * cos(eta) + theta_dot * cos(phi) * sin(eta) + phi_dot * (cos(theta) * sin(eta) * sin(phi) - cos(eta) * sin(theta)) + ...
    phi_dot * theta_dot * (cos(eta) * cos(phi) * sin(phi) - sin(eta) - (cos(phi)^2) * sin(eta))) + ...
    phi_dot * psi_dot * (sin(eta) * sin(phi) - cos(eta) * cos(phi)) * cos(theta) * cos(phi) + ...
    phi_dot * theta_dot * (sin(phi) * sin(theta) * (1 + cos(eta)) - cos(theta) * sin(eta) * (1 + sin(eta))) * cos(eta) * cos(phi);

  alpha_dist_Oz = -theta_ddot * sin(phi) + psi_ddot * cos(phi) * cos(theta) + ...
    psi_dot * theta_dot * (cos(phi) * cos(theta) * sin(eta) * sin(phi) - sin(phi)^2 * sin(theta) + cos(eta) * cos(phi)^2 * sin(theta)) + ...
    phi_dot * (psi_dot * (cos(phi) * cos(theta) * sin(phi) - sin(phi) * sin(theta)) + theta_dot * cos(phi)^2);

  % A.4
  temp1 = phi_ddot * cos(eps) * cos(eta) + theta_ddot * (sin(eps) * sin(phi) + cos(eps) * cos(phi) * sin(eta));
  temp2 = psi_ddot * (cos(eps) * cos(eta) * sin(theta) - cos(phi) * cos(theta) * sin(eps) + cos(eps) * cos(theta) * sin(eta) * sin(phi));
  temp3 = eta_dot * eps_dot * cos(eps) - phi_dot * theta_dot * cos(phi)^2 * sin(eps) + phi_dot * theta_dot * cos(eps) * cos(eta) + eta_dot * phi_dot * cos(eps) * sin(eta) + eps_dot * phi_dot * cos(eta) * sin(eps);
  temp4 = -eps_dot * theta_dot * cos(eps) * sin(phi) + psi_dot * theta_dot * sin(eps) * sin(phi)^2 * sin(theta) - eta_dot * theta_dot * cos(eps) * cos(eta) * cos(phi) + eps_dot * psi_dot * cos(eps) * cos(phi) * cos(theta);
  temp5 = eps_dot * theta_dot * cos(phi) * sin(eps) * sin(eta) - eta_dot * psi_dot * cos(eps) * sin(eta) * sin(theta) - eps_dot * psi_dot * cos(eta) * sin(eps) * sin(theta) + phi_dot * psi_dot * sin(eps) * sin(phi) * sin(theta);
  temp6 = -phi_dot * theta_dot * cos(eps) * cos(eta) * cos(phi)^2 - psi_dot * theta_dot * cos(eps) * cos(phi)^2 * cos(theta) * sin(eta)^2 - eta_dot * psi_dot * cos(eps) * cos(eta) * cos(theta) * sin(phi);
  temp7 = phi_dot * theta_dot * cos(eps) * cos(phi) * sin(eta) * sin(phi) - phi_dot * psi_dot * cos(phi) * cos(theta) * sin(eps) * sin(phi) + eps_dot * psi_dot * cos(theta) * sin(eps) * sin(eta) * sin(phi);
  temp8 = psi_dot * theta_dot * cos(eps) * cos(eta)^2 * cos(phi) * cos(theta) - phi_dot * psi_dot * cos(eps) * cos(phi)^2 * cos(theta) * sin(eta) - psi_dot * theta_dot * cos(eta) * cos(phi)^2 * sin(eps) * sin(theta);
  temp9 = psi_dot * theta_dot * (cos(eps) * sin(theta) - cos(theta) * sin(eps) + cos(eps) * cos(eta) * sin(theta)) * cos(phi) * sin(phi) * sin(eta);
  temp10 = -phi_dot * psi_dot * cos(eps) * cos(eta) * cos(phi) * cos(theta) * sin(phi);
  alpha_dist_Ix = temp1 - temp2 + temp3 - temp4 + temp5 - temp6 + temp7 + temp8 + temp9 - temp10;

  % A.5
  alpha_dist_Iy = -phi_ddot * sin(eta) + psi_ddot * (cos(eta) * cos(theta) * sin(phi) + sin(eta) * sin(theta)) + theta_ddot * cos(eta) * cos(phi) - ...
    phi_dot * theta_dot * sin(eta) + eta_dot * phi_dot * cos(eta) + phi_dot * theta_dot * cos(phi)^2 * sin(eta) + psi_dot * theta_dot * cos(eta)^2 * cos(phi) * sin(phi) * sin(theta) + ...
    eta_dot * theta_dot * cos(phi) * sin(eta) - eta_dot * psi_dot * cos(eta) * sin(theta) + phi_dot * theta_dot * cos(eta) * cos(phi) * sin(phi) + ...
    eta_dot * (psi_dot * cos(theta) * sin(eta) * sin(phi) - phi_dot * psi_dot * cos(eta) * cos(phi)^2 * cos(theta) - psi_dot * theta_dot * cos(eta) * cos(phi) * cos(theta) * sin(eta) + ...
    phi_dot * psi_dot * cos(phi) * cos(theta) * sin(eta) * sin(phi) + psi_dot * theta_dot * cos(eta) * cos(phi) * sin(phi) * sin(theta)) - ...
    psi_dot * theta_dot * cos(eta) * cos(phi)^2 * cos(theta) * sin(eta);

  % A.6
  temp1 = phi_ddot * cos(eta) * sin(eps) + theta_ddot * (cos(phi) * sin(eps) * sin(eta) - cos(eps) * sin(phi));
  temp2 = psi_ddot * (cos(eps) * cos(phi) * cos(theta) - cos(eta) * sin(eps) * sin(theta) + cos(theta) * sin(eps) * sin(eta) * sin(phi));
  temp3 = eta_dot * (eps_dot + phi_dot * sin(eta) - psi_dot * sin(eta) * sin(theta) - psi_dot * cos(eta) * cos(theta) * sin(phi) - theta_dot * cos(eta) * cos(phi)) * sin(eps);
  temp4 = eps_dot * psi_dot * (cos(eps) * cos(eta) * sin(theta) + cos(phi) * cos(theta) * sin(eps) - cos(eps) * cos(theta) * sin(eta) * sin(phi));
  temp5 = -eps_dot * phi_dot * (cos(eps) * cos(eta) - eps_dot * theta_dot * (sin(eps) * sin(phi) - cos(eps) * cos(phi) * sin(eta)));
  temp6 = phi_dot * theta_dot * (cos(phi) * sin(eps) * sin(eps) * sin(eta) * sin(phi) + cos(eps) * cos(phi)^2 + cos(eta) * sin(eps) - cos(eta) * cos(phi)^2 * sin(eps));
  temp7 = psi_dot * theta_dot * (cos(eps) * cos(eta) * cos(phi)^2 * sin(theta) + cos(eta)^2 * cos(phi) * cos(theta) * sin(eps) - cos(eps) * sin(phi)^2 * sin(theta));
  temp8 = -cos(phi)^2 * cos(theta) * sin(eps) * sin(eta)^2 + cos(eps) * cos(phi) * cos(theta) * sin(eta) * sin(phi);
  temp9 = cos(phi) * sin(eps) * sin(eta) * sin(phi) * sin(theta) + cos(eta) * cos(phi) * sin(eps) * sin(eta) * sin(phi) * sin(theta);
  temp10 = phi_dot * psi_dot * (cos(eps) * cos(phi) * cos(theta) * sin(phi) - cos(eps) * sin(phi) * sin(theta) - cos(phi)^2 * cos(theta) * sin(eps) * sin(eta));
  temp11 = -cos(eta) * cos(phi) * cos(theta) * sin(eps) * sin(phi);
  alpha_dist_Iz = temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp6 + temp7 + temp8 + temp9 + temp10 + temp11;

  alphas = [alpha_O_ex; alpha_O_ey; alpha_dist_Ix; alpha_dist_Iy; alpha_dist_Iz; alpha_dist_Oz];
end
