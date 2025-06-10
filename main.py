from models.double_a_arm import DoubleAArm, DoubleAArmHardpoints

if __name__ == "__main__":
    hp = DoubleAArmHardpoints.from_yml("hardpoints.yml")
    solver = DoubleAArm(hp)

    travels_ref = [
        [133.000, 699.611, 304.801],
        [119.152, 669.778, 222.537],
        [153.431, 715.172, 423.216],
    ]
    travels = [
        1e-4,       # static
        -81,        # bump
        122,        # droop
    ]
    travels_space = [i for i in range(travels[1], travels[2], 1)]
    travels_space.remove(0)

    steers = [
        1e-4,      # static
        -25,       # left
        25,        # right
    ]
    steers_space = [i for i in range(steers[1], steers[2], 1)]

    data = []
    for travel in travels_space:
        for steer in steers_space:
            out = solver.solve(travel, steer=steer)
            data.append(out.wc)

    # plot the results in 3d
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*zip(*data), c='r', marker='o', s=5)
    ax.scatter(*zip(*travels_ref), c='b', marker='x', s=100)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Double A-Arm Suspension Kinematics')
    ax.legend(['Calculated', 'Reference'])
    plt.show()