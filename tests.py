from utils import *
from base import *
from models import *




if __name__ == "__main__":
    N = 20
    T = 7

    radius = 1
    speed = 6
    target = np.array([0,0])
    safe_target = 1.5
    safe_drone = 3

    #fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    cm = IAM(N, T, InterOrbit, [target, radius, speed, safe_target, safe_drone], seed=1)
    cm.solve_ivp()
    #cm.plot_state(cm.solution_positions[-1], cm.solution_velocities[-1], ax=axs[1])
    #axs[0].set_title(f'Initial Condition for beta = {beta}')
    #axs[1].set_title(f'Final Condition for beta = {beta}')
    #plt.tight_layout()
    #plt.show()
    
    cm.animate_solution_2(interval=10)
    #import code
    #code.interact(local=locals())  # Interactive mode
#+ self.repulsion(r[i], v[i]) + self.speed_adjust(v[i])