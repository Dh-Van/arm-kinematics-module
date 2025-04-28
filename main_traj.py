from modules.trajectory_generator import *


def main():
    """Main function that runs the simulation"""

    traj = MultiAxisTrajectoryGenerator(method="quintic",
                                        mode="task",
                                        interval=[0,10],
                                        ndof=3,
                                        start_pos=[0, 0, 0],
                                        final_pos=[10, 20, -10])
    
    # generate trajectory
    t = traj.generate(nsteps=30)

    # plot trajectory
    traj.plot()


if __name__ == "__main__":
    main()