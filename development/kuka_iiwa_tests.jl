using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO
using MeshCatMechanisms

traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics/urdf")
urdf = joinpath(urdf_folder, "kuka_iiwa.urdf")

mechanism = parse_urdf(urdf)

mvis = MechanismVisualizer(mechanism, URDFVisuals(urdf))
open(mvis)

float64state = MechanismState(mechanism)
rand!(float64state)

q = configuration(float64state)
v = velocity(float64state)

set_configuration!(float64state, q)
set_velocity!(float64state, v)
set_configuration!(mvis, configuration(float64state))
