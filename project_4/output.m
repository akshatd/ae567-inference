% Nonlinear Output Equations
function obs = output(x_G, u_G, w_G)
  y1 = x_G(1);
  y2 = x_G(2);
  y3 = x_G(3) - w_G(4) * sin(w_G(2)) + w_G(3) * cos(w_G(2)) * cos(w_G(1));
  y4 = x_G(4) - w_G(5) * sin(w_G(1)) + w_G(4) * cos(x_G(1)) * cos(w_G(2)) + ...
    w_G(3) * cos(x_G(1)) * sin(w_G(2));
  obs = [y1; y2; y3; y4];
end
