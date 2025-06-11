from models.double_a_arm_dixon import DixonDoubleAArm, DoubleAArmHardpoints

if __name__ == "__main__":
    hp = DoubleAArmHardpoints.from_yml("hardpoints.yml")
    solver = DixonDoubleAArm(hp)

    out = solver.front_view_analysis(hp, 80)
    out = solver.front_view_analysis(hp, -120)