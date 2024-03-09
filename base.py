
from utils import *

# A generic class for Interacting Agent models
# The specific model is defined in another class that
# is passed as a parameter.
class IAM:
  def __init__(self, N_particles, T, Model, params, dt=0.01, seed=None):
    self.N = N_particles
    self.T = T
    self.dt = dt
    self.params = params
    self.t = np.arange(0,T,dt)
    self.model = Model(self)

    if seed is not None:
      np.random.seed(seed)

    self.initialize()

  def initialize(self):
    self.positions = np.random.uniform(-2,2,[self.N, 2])
    self.random_angles = np.random.uniform(0,2*np.pi, self.N)
    self.velocities = np.array([[np.cos(angle), np.sin(angle)] for angle in self.random_angles])
    self.state0_flat = np.concatenate([self.positions.flatten(), self.velocities.flatten()])


  def system(self, state_flat, t):
    N = len(state_flat) // 4
    y = state_flat.reshape(2*N, 2)
    r = state_flat[:2*N].reshape(N, 2)
    v = state_flat[2*N:].reshape(N, 2)
    drdt = self.model.dr_dt(v)
    dvdt = self.model.dv_dt(r, v)
    return np.concatenate([drdt.flatten(), dvdt.flatten()])


  def solve(self):
    solution_flat = odeint(self.system, self.state0_flat, self.t)

    time_steps = len(self.t)
    self.solution_positions = []
    self.solution_velocities = []
    for t in range(time_steps):
      reshaped = solution_flat[t].reshape(2*(self.N), 2)
      positions = reshaped[:self.N, :]
      velocities = reshaped[self.N:, :]

      self.solution_positions.append(positions)
      self.solution_velocities.append(velocities)



  def system_ivp(self, t, state_flat):
    N = self.N
    y = state_flat.reshape(2*N, 2)
    r = y[:N, :]
    v = y[N:, :]
    drdt = self.model.dr_dt(v)
    dvdt = self.model.dv_dt(r, v)
    return np.concatenate([drdt.flatten(), dvdt.flatten()])

  def solve_ivp(self):
    sol = solve_ivp(self.system_ivp, [0, self.T], self.state0_flat, method='RK45', t_eval=self.t, vectorized=False)

    time_steps = len(self.t)
    self.solution_positions = []
    self.solution_velocities = []

    for t in range(time_steps):
      reshaped = sol.y[:, t].reshape(2*self.N, 2)
      positions = reshaped[:self.N, :]
      velocities = reshaped[self.N:, :]

      self.solution_positions.append(positions)
      self.solution_velocities.append(velocities)

  # Perhaps a Milstein would be more appropriated
  def solve_euler_maruyama(self):
    time_steps = len(self.t)
    db = np.random.normal(0, np.sqrt(self.dt), [time_steps]) #, self.N, 1])
    self.solution_positions = [self.positions]
    self.solution_velocities = [self.velocities]
    posi_0 = self.positions
    velo_0 = self.velocities
    for t in range(time_steps):
      dr = self.model.dr(velo_0)*self.dt
      posi_1 = posi_0 + dr
      dv = (self.model.dv_det(posi_0, velo_0)*self.dt +
                self.model.dv_stoch(posi_0, velo_0)*db[t])
      #print(dv)
      velo_1 = velo_0 + dv


      self.solution_positions.append(posi_1)
      self.solution_velocities.append(velo_1)
      posi_0 = posi_1
      velo_0 = velo_1


  def mean_velocity_(self):
    self.mean_velocity = np.array([np.mean(self.solution_velocities[i], axis=0) for i in range(len(self.t))])
    return self.mean_velocity

  def order_param_(self):
    self.order_param = np.array([np.mean(np.linalg.norm(self.solution_velocities[i]-self.mean_velocity[i], axis=1)**2) for i in range(len(self.t))])
    return self.order_param

  def plot_state(self, positions=None , velocities=None, ax=None):
    if (positions is None) or (velocities is None):
      positions = self.positions
      velocities = self.velocities

    x_min = positions.T[0].min()-0.2
    x_max = positions.T[0].max()+0.2
    y_min = positions.T[1].min()-0.2
    y_max = positions.T[1].max()+0.2
    x_range = x_max - x_min
    y_range = y_max - y_min

    if ax is None:
        fig, ax = plt.subplots()
        ax_created = True
    else:
        ax_created = False

    ax.scatter(positions[:, 0], positions[:, 1], color='tab:blue')
    
    for position, velocity in zip(positions, velocities):
      ax.arrow(position[0], position[1],
                velocity[0]/np.linalg.norm(velocity)*0.05*x_range,
                velocity[1]/np.linalg.norm(velocity)*0.05*y_range,
                head_width=0.01*x_range, head_length=0.02*y_range, fc='tab:orange', ec='tab:orange')



    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Positions and Velocities')
    if ax_created:
        plt.show()




  def __getattr__(self, name):
    # Method is called if attribute isn't found
    if name == "mean_velocity":
      return self.mean_velocity_()
    elif name == "order_param":
      return self.order_param_()
    elif name == "solution_positions":
      return self.solve()
    elif name == "solution_velocities":
      return self.solve()
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


  def animate_solution_2(self, interval=20, save=False):
    fig, ax = plt.subplots()
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    points = ax.scatter(self.solution_positions[0][:, 0], self.solution_positions[0][:, 1], color='blue')
    center = ax.scatter(0, 0, color='red')
    # Initial arrows (empty, will be updated)
    arrows = [ax.arrow(self.solution_positions[0][_][0], self.solution_positions[0][_][1],
                       self.solution_velocities[0][_][0]*0.05, self.solution_velocities[0][_][1]*0.05,
                       head_width=0.1, head_length=0.15, fc='tab:orange', ec='tab:orange') for _ in range(self.N)]

    def update(frame):
      # Update points positions
      points.set_offsets(self.solution_positions[frame])
      # Update arrows (velocities)
      for i, arrow in enumerate(arrows):
          arrow.set_visible(False)  # Hide the old arrow
          # Create new arrow
          arrows[i] = ax.arrow(self.solution_positions[frame][i, 0], self.solution_positions[frame][i, 1],
                              self.solution_velocities[frame][i, 0]*0.05, self.solution_velocities[frame][i, 1]*0.05,
                              head_width=0.1, head_length=0.15, fc='tab:orange', ec='tab:orange')
      return center, points, *arrows

    

    if save:
      anim = FuncAnimation(fig, update, frames=range(len(self.solution_positions)), interval=1, blit=True)
      anim.save('animation.gif', writer='pillow', fps=60)
    else:
      anim = FuncAnimation(fig, update, frames=range(len(self.solution_positions)), interval=interval, blit=True)
      plt.tight_layout()
      plt.show()
    #HTML(anim.to_html5_video())


  def animate_solution(self):
      fig, ax = plt.subplots()
      ax.set_xlim(-1, 1)
      ax.set_ylim(-1, 1)


      points = ax.scatter(self.solution_positions[0][:, 0], self.solution_positions[0][:, 1], color='blue')


      arrows = [ax.arrow(self.solution_positions[0][_][0], self.solution_positions[0][_][1],
                        self.solution_velocities[0][_][0]*0.05, self.solution_velocities[0][_][1]*0.05,
                        head_width=0.01, head_length=0.02, fc='tab:orange', ec='tab:orange') for _ in range(self.N)]

      def update(frame):

          points.set_offsets(self.solution_positions[frame])
          for arrow in arrows:
              arrow.remove()
          arrows[:] = [ax.arrow(self.solution_positions[frame][i, 0], self.solution_positions[frame][i, 1],
                                self.solution_velocities[frame][i, 0]*0.05, self.solution_velocities[frame][i, 1]*0.05,
                                head_width=0.01, head_length=0.02, fc='tab:orange', ec='tab:orange') for i in range(self.N)]
          return points, *arrows

      return FuncAnimation(fig, update, frames=range(len(self.solution_positions)), interval=20, blit=True)
