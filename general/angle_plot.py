import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

file = np.loadtxt('output.out')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#turns cartesian coordinates and a rotation angle into a rotation quarternion
def complete_quartenion(rotation_angle, vector):
    if (vector == [0,0,0]):
        return [1,0,0,0]
    else:
        norm = np.linalg.norm(vector)
        w = math.cos(rotation_angle/2)
        x = (vector[0]/norm)* math.sin(rotation_angle/2)
        y = (vector[1]/norm)* math.sin(rotation_angle/2)
        z = (vector[2]/norm)* math.sin(rotation_angle/2)

    return [w,x,y,z]

    







def file_line_to_point(file_line_arr):
    point = [file_line_arr[0], file_line_arr[1], file_line_arr[2]]
    return point

def file_line_to_quarternion(file_line_arr):
    quarternion = [file_line_arr[6], file_line_arr[3], file_line_arr[4], file_line_arr[4]]
    return quarternion #puts this in [qw,qx,qy,qz]

def initial_point_quartenion(point_coordinates):
    point_coordinates = np.array(point_coordinates)
    # norm = np.linalg.norm(point_coordinates)
    # base_quarternion = [0, point_coordinates[0]/norm, point_coordinates[1]/norm, point_coordinates[2]/norm]
    base_quarternion = [0, point_coordinates[0], point_coordinates[1], point_coordinates[2]] # do not turn into unit vector
    return base_quarternion # returns as [qw, qx, qy, qz]

def rotation_quarternion_inverse(quarternion):
    return [quarternion[0], -quarternion[1], -quarternion[2], -quarternion[3]]

def rotated_cartesian_point(final_quarternion):
    return [final_quarternion[1], final_quarternion[2], final_quarternion[3]]

def quarternion_multiplication(quarternion1,  quarternion2):
    t0 = quarternion1[0]*quarternion2[0] - quarternion1[1]*quarternion2[1] - quarternion1[2]*quarternion2[2] - quarternion1[3]*quarternion2[3]
    t1 = quarternion1[0]*quarternion2[1] + quarternion1[1]*quarternion2[0] - quarternion1[2]*quarternion2[3] + quarternion1[3]*quarternion2[2]
    t2 = quarternion1[0]*quarternion2[2] + quarternion1[1]*quarternion2[3] + quarternion1[2]*quarternion2[0] - quarternion1[3]*quarternion2[1]
    t3 = quarternion1[0]*quarternion2[3] - quarternion1[1]*quarternion2[2] + quarternion1[2]*quarternion2[1] + quarternion1[3]*quarternion2[0]
    return [t0,t1,t2,t3]

def active_rotation(q_inverse,initial_point_quartenion, rotation_quarternion):
    #p' = (q^-1)(p)(q)
    left_product = quarternion_multiplication(q_inverse, initial_point_quartenion)
    new_point = quarternion_multiplication(left_product, rotation_quarternion)

    return new_point

def passive_rotation (q_inverse,initial_point_quartenion, quarternion):
     #p' = (q)(p)(q^-1)
    left_product = quarternion_multiplication(quarternion, initial_point_quartenion)
    new_point = quarternion_multiplication(left_product, quarternion)
    return new_point

def rotate_body(np_arr, rotation_quarternion):
    rotated = []
    q_inverse = rotation_quarternion_inverse(rotation_quarternion)
    for points in np_arr:
        
        quarternion_points =  initial_point_quartenion(points)
       
        rotated_points = active_rotation(q_inverse, quarternion_points, rotation_quarternion)
        final_rotated_points = rotated_cartesian_point(rotated_points)
        rotated.append(final_rotated_points)

    return (np.array(rotated)) 
    
def center_of_mass (np_arr):
    x = 0
    y = 0
    z = 0
    for point in np_arr:
        x += point[0]
        y += point[1]
        z += point[2]
    
    center_of_mass = [x/len(np_arr),y/len(np_arr),z/len(np_arr)]
    return np.array(center_of_mass)

def shift_body(np_arr, shift):
    final_points = np_arr + shift

    return final_points

#creates a cube of length 1 from the center of mass 
def create_cube_from_com(center_of_mass):
    d = 0.5
    return np.array([[center_of_mass[0] - d, center_of_mass[1]+ d,center_of_mass[2] - d],
                     [center_of_mass[0] + d, center_of_mass[1] - d,center_of_mass[2] - d],
                     [center_of_mass[0] - d, center_of_mass[1] - d,center_of_mass[2] - d],
                     [center_of_mass[0] + d, center_of_mass[1] - d,center_of_mass[2] + d],
                     [center_of_mass[0] - d, center_of_mass[1] - d,center_of_mass[2] + d],
                     [center_of_mass[0] - d, center_of_mass[1] + d,center_of_mass[2] + d],
                     [center_of_mass[0] + d, center_of_mass[1] + d,center_of_mass[2] + d],              
                     ])

def plot_cube_quarternion (center_of_mass, qw, qx,qy, qz,plot):
    rotation_quarternion = [qw, qx, qy , qz]
    print(rotation_quarternion)
    # rotate the cube when it is centered at the origin
    cube_points_rotated = rotate_body(origin_cube_points, rotation_quarternion) 
    # shift the cube from the origin to the center of mass
    final_cube =  shift_body(cube_points_rotated,center_of_mass)
    cube_plot(plot,final_cube[:, 0], final_cube[:, 1], final_cube[:, 2],)
    return final_cube


def plot_cube_cartesian (center_of_mass, axis, theta,plot):
    rotation_quarternion = complete_quartenion(theta, axis) 
    # rotate the cube when it is centered at the origin
    cube_points_rotated = rotate_body(origin_cube_points, rotation_quarternion) 
    # shift the cube from the origin to the center of mass
    final_cube =  shift_body(cube_points_rotated,center_of_mass)
    cube_plot(plot,final_cube[:, 0], final_cube[:, 1], final_cube[:, 2],)
    return final_cube

def cube_plot(plot, x_array, y_array, z_array,):
    # Face IDs
    vertices = [[0,1,3,2],[1,0,7,6],[1,6,5,3],[2,3,5,4],[0,2,4,7],[4,5,6,7]]
    tupleList = list(zip(x_array, y_array, z_array)) 
    plot.scatter3D(x_array, y_array, z_array)
    #Set up cube surface 
    poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
    plot.add_collection3d(Poly3DCollection(poly3d, facecolors='r', linewidths=1, alpha=0.5))

cube_points = np.array([[1.5,1.5, 0.5],
                        [0.5,1.5,0.5],
                        [1.5,0.5,0.5],
                        [0.5,0.5,0.5],
                        [1.5,0.5,1.5],
                        [0.5,0.5,1.5],
                        [0.5,1.5,1.5],
                        [1.5,1.5,1.5]])

origin_cube_points = np.array([[0.5,0.5, -0.5],
                        [-0.5,0.5,-0.5],
                        [0.5,-0.5,-0.5],
                        [-0.5,-0.5,-0.5],
                        [0.5,-0.5,0.5],
                        [-0.5,-0.5,0.5],
                        [-0.5,0.5,0.5],
                        [0.5,0.5,0.5]])


# rotation_quarternion = complete_quartenion(math.pi/4, [1,0,0]) 
# print(f"rotation quarternion around the origin: {rotation_quarternion}")

# com = center_of_mass(cube_points)
# print(f"center of mass {com}")
## shift everypoint of the cube so that the COM is at the origin
# shifted_np_array = shift_body(cube_points, -(com)) 
# ax.scatter3D(shifted_np_array[:,0],shifted_np_array[:,1],shifted_np_array[:,2])

# print(center_of_mass(shifted_np_array))
# print(f"points shifted to origin: {shifted_np_array}")

# cube_points_rotated = rotate_body(shifted_np_array, rotation_quarternion) 

# print(f"rotated cube np array: {cube_points_rotated}")
# ax.scatter3D(cube_points_rotated[:,0],cube_points_rotated[:,1],cube_points_rotated[:,2])
# print(center_of_mass(cube_points_rotated))
# cube_points_rotated = np.array(cube_points_rotated)

#shift every rotated point by the shift vector 
# cube_points_shifted =  shift_body(cube_points_rotated,com)
# print(f"final cube  {cube_points_shifted}")

# x_array = cube_points_shifted[:, 0]
# y_array = cube_points_shifted[:, 1]
# z_array = cube_points_shifted[:, 2]
# cube_plot(ax,x_array, y_array, z_array, cube_points_shifted)

test_com =[file[80][0], file[80][1], file[80][2]] #xyz for com
print(f"test com {test_com}")
test_qw = file[80][6]
test_qx = file[80][3]
test_qy = file[80][4]
test_qz = file[80][5]

rotate_test = plot_cube_quarternion(test_com, test_qw, test_qx,test_qy,test_qz,ax)

test_com1 =[file[5732][0], file[5732][1], file[5732][2]] #xyz for com
print(f"test com1 {test_com}")
test_qw1 = file[5732][6]
test_qx1 = file[5732][3]
test_qy1 = file[5732][4]
test_qz1 = file[5732][5]

rotate_test1 = plot_cube_quarternion(test_com1, test_qw1, test_qx1,test_qy1,test_qz1,ax)
print(rotate_test1)

new_com1 = center_of_mass(rotate_test1)
print(new_com1)

plt.show()






