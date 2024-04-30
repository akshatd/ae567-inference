%% Plotting function
function plot_sys(data)
  plot_states(data.x, data.t)
  plot_control(data.u, data.t)
  plot_dist(data.w, data.t)
end

function plot_states(states, t)
  figure()
  subplot(3, 2, 1);
  plot(t, states(:, 1), 'b', 'LineWidth', 2, DisplayName = "x_{G1}");
  hold on
  yline(-pi, '--r', 'LineWidth', 2, DisplayName = "lower limit");
  yline(pi, '--r', 'LineWidth', 2, DisplayName = "upper limit");
  xlabel('Time (s)');
  ylabel('$\eta$ (rad)', 'Interpreter', 'latex', FontSize = 16);
  yline(1, '--k', DisplayName = "reference")
  grid on; grid minor
  legend show

  subplot(3, 2, 2);
  plot(t, states(:, 2), 'b', 'LineWidth', 2, DisplayName = "x_{G2}");
  hold on
  yline(-pi / 9, '--r', 'LineWidth', 2, DisplayName = "lower limit");
  yline(4 * pi / 9, '--r', 'LineWidth', 2, DisplayName = "upper limit");
  xlabel('Time (s)');
  ylabel('$\epsilon$ (rad)', 'Interpreter', 'latex', FontSize = 16);
  yline(0.25, '--k', DisplayName = "reference")
  grid on; grid minor
  legend show

  subplot(3, 2, 3);
  plot(t, states(:, 3), 'b', 'LineWidth', 2, DisplayName = "x_{G3}");
  hold on
  yline(-20, '--r', 'LineWidth', 2, DisplayName = "lower limit");
  yline(20, '--r', 'LineWidth', 2, DisplayName = "upper limit");
  xlabel('Time (s)');
  ylabel('$\Omega_{OBz}$ (rad/sec)', 'Interpreter', 'latex', FontSize = 16);
  grid on; grid minor
  legend show

  subplot(3, 2, 4);
  plot(t, states(:, 4), 'b', 'LineWidth', 2, DisplayName = "x_{G4}");
  hold on
  yline(-12, '--r', 'LineWidth', 2, DisplayName = "lower limit");
  yline(12, '--r', 'LineWidth', 2, DisplayName = "upper limit");
  xlabel('Time (s)');
  ylabel('$\Omega_{IOy}$ (rad/sec)', 'Interpreter', 'latex', FontSize = 16);
  grid on; grid minor
  legend show

  subplot(3, 2, 5);
  plot(t, states(:, 5), 'b', 'LineWidth', 2, DisplayName = "x_{G5}");
  hold on
  yline(-10, '--r', 'LineWidth', 2, DisplayName = "lower limit");
  yline(10, '--r', 'LineWidth', 2, DisplayName = "upper limit");
  xlabel('Time (s)');
  ylabel('$z_{OB}$', 'Interpreter', 'latex', FontSize = 16);
  grid on; grid minor
  legend show

  subplot(3, 2, 6);
  plot(t, states(:, 6), 'b', 'LineWidth', 2, DisplayName = "x_{G6}");
  hold on
  yline(-10, '--r', 'LineWidth', 2, DisplayName = "lower limit");
  yline(10, '--r', 'LineWidth', 2, DisplayName = "upper limit");
  xlabel('Time (s)');
  ylabel('$z_{IO}$', 'Interpreter', 'latex', FontSize = 16);
  grid on; grid minor
  legend show

  sgtitle('Gimbal States');
end

function plot_control(control, t)
  figure()
  subplot(2, 1, 1);
  plot(t, control(:, 1), 'b', 'LineWidth', 2, DisplayName = "u_{G1}");
  hold on
  yline(-203, '--r', 'LineWidth', 2, DisplayName = "lower limit");
  yline(203, '--r', 'LineWidth', 2, DisplayName = "upper limit");
  xlabel('Time (s)');
  ylabel('$T_{mO}$ (Nm)', 'Interpreter', 'latex', FontSize = 16);
  grid on; grid minor
  legend show

  subplot(2, 1, 2);
  plot(t, control(:, 2), 'b', 'LineWidth', 2, DisplayName = "u_{G2}");
  hold on
  yline(-35.3, '--r', 'LineWidth', 2, DisplayName = "lower limit");
  yline(35.3, '--r', 'LineWidth', 2, DisplayName = "upper limit");
  xlabel('Time (s)');
  ylabel('$T_{mI}$ (Nm)', 'Interpreter', 'latex', FontSize = 16);
  grid on; grid minor
  legend show

  sgtitle("Control")
end

function plot_dist(dist, t)
  figure()
  subplot(4, 2, 1);
  plot(t, dist(:, 1), '-b', 'LineWidth', 2, 'DisplayName', '\theta');
  hold on
  yline(-pi / 20, '--r', 'LineWidth', 2, 'DisplayName', 'lower limit');
  yline(pi / 20, '--r', 'LineWidth', 2, 'DisplayName', 'upper limit');
  xlabel('Time (s)', 'Interpreter', 'latex');
  ylabel('$\theta$ (rad)', 'Interpreter', 'latex', 'FontSize', 16);
  grid on; grid minor
  legend show

  subplot(4, 2, 2);
  plot(t, dist(:, 2), '-b', 'LineWidth', 2, 'DisplayName', '\phi');
  hold on
  yline(-pi / 8, '--r', 'LineWidth', 2, 'DisplayName', 'lower limit');
  yline(pi / 8, '--r', 'LineWidth', 2, 'DisplayName', 'upper limit');
  xlabel('Time (s)', 'Interpreter', 'latex');
  ylabel('$\phi$ (rad)', 'Interpreter', 'latex', 'FontSize', 16);
  grid on; grid minor
  legend show

  subplot(4, 2, 3);
  plot(t, dist(:, 3), 'b', 'LineWidth', 2, 'DisplayName', '\psi dot');
  hold on
  yline(-pi / 180, '--r', 'LineWidth', 2, 'DisplayName', 'lower limit');
  yline(pi / 180, '--r', 'LineWidth', 2, 'DisplayName', 'upper limit');
  xlabel('Time (s)', 'Interpreter', 'latex');
  ylabel('$\dot \psi$ (rad/sec)', 'Interpreter', 'latex', 'FontSize', 16);
  grid on; grid minor
  legend show

  subplot(4, 2, 4);
  plot(t, dist(:, 4), '-b', 'LineWidth', 2, 'DisplayName', '\theta dot');
  hold on
  yline(-pi / 60, '--r', 'LineWidth', 2, 'DisplayName', 'lower limit');
  yline(pi / 60, '--r', 'LineWidth', 2, 'DisplayName', 'upper limit');
  xlabel('Time (s)', 'Interpreter', 'latex');
  ylabel('$\dot \theta$ (rad/sec)', 'Interpreter', 'latex', 'FontSize', 16);
  grid on; grid minor
  legend show

  subplot(4, 2, 5);
  plot(t, dist(:, 5), 'b', 'LineWidth', 2, 'DisplayName', '\phi dot');
  hold on
  yline(-pi / 18, '--r', 'LineWidth', 2, 'DisplayName', 'lower limit');
  yline(pi / 18, '--r', 'LineWidth', 2, 'DisplayName', 'upper limit');
  xlabel('Time (s)', 'Interpreter', 'latex');
  ylabel('$\dot \phi$ (rad/sec)', 'Interpreter', 'latex', 'FontSize', 16);
  grid on; grid minor
  legend show

  subplot(4, 2, 6);
  plot(t, dist(:, 6), 'b', 'LineWidth', 2, 'DisplayName', '\psi ddot');
  hold on
  yline(-pi / 180, '--r', 'LineWidth', 2, 'DisplayName', 'lower limit');
  yline(pi / 180, '--r', 'LineWidth', 2, 'DisplayName', 'upper limit');
  xlabel('Time (s)', 'Interpreter', 'latex');
  ylabel('$\ddot \psi (rad/sec^2)$', 'Interpreter', 'latex', 'FontSize', 16);
  grid on; grid minor
  legend show

  subplot(4, 2, 7);
  plot(t, dist(:, 7), '-b', 'LineWidth', 2, 'DisplayName', '\theta ddot');
  hold on
  yline(-pi / 60, '--r', 'LineWidth', 2, 'DisplayName', 'lower limit');
  yline(pi / 60, '--r', 'LineWidth', 2, 'DisplayName', 'upper limit');
  xlabel('Time (s)', 'Interpreter', 'latex');
  ylabel('$\ddot \theta (rad/sec^2)$', 'Interpreter', 'latex', 'FontSize', 16);
  grid on; grid minor
  legend show

  subplot(4, 2, 8);
  plot(t, dist(:, 8), 'b', 'LineWidth', 2, 'DisplayName', '\phi ddot');
  hold on
  yline(-pi / 18, '--r', 'LineWidth', 2, 'DisplayName', 'lower limit');
  yline(pi / 18, '--r', 'LineWidth', 2, 'DisplayName', 'upper limit');
  xlabel('Time (s)', 'Interpreter', 'latex');
  ylabel('$\ddot \phi (rad/sec^2)$', 'Interpreter', 'latex', 'FontSize', 16);
  grid on; grid minor
  legend show

  sgtitle('Disturbances');
end
