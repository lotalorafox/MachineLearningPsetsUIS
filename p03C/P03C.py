import numpy as np
def move_to_target( particles ):
        
    target_point = [10,50]
    sx, sy = np.std(np.array([[p.x,p.y] for p in particles]), axis=0)
    mx, my, o = np.mean(np.array([[p.x,p.y, p.orientation] for p in particles]), axis=0)

    turn, forward = 0,0
    # calculate the angle 
    turn = -o
    x = target_point[0]-mx
    y = target_point[1]-my
    d = np.sqrt(((x**2)+(y**2))) 
    sea = y/h
    alfa = np.arcsin(sea)
    turn = turn + alfa
    forward = d
    return turn, forward


lmarks  = [[20.0, 20.0], [80.0, 80.0], [20.0, 80.0], [80.0, 20.0]]#, [50., 50.], [50., 20.]]
wsize = 100.0
robot_positions, particles, errs_mean, errs_std, parts_std = trayectory(n_particles=1000, max_steps=200, 
                                                                        landmarks=lmarks, world_size=wsize,
                                                                        next_move=move_to_target, 
                                                                        init_pos = [95,5])
plt.figure()
plot_all(wsize, lmarks, robot_positions, particles, errs_mean, errs_std, parts_std)