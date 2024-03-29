import numpy as np 
import math
#roll (x-axis rotation)
#pitch (y-axis rotation)
#yaw (z-axis rotation)
file = np.loadtxt('output.out')
positive_gimbal_lock_cuttoff = 0.4999
negative_gimbal_lock_cuttoff = -0.4999

print (file[40][3])


#initialize and empty list of 4 zeros since quarternions only have 4 values
quarternions = np.zeros(4) #order of the values is [x,y,z,w]

#testing on 1 set of quarternions
for i in range (4):
    quarternions[i] = file[40][i+3]

print (quarternions) 

## calculate components of roll, pitch, yaw
w_squared = quarternions[3]*quarternions[3]
x_squared = quarternions[0]*quarternions[0]
y_squared = quarternions[1]*quarternions[1]
z_squared = quarternions[2]*quarternions[2]

#unit vector check
unit_vector = w_squared + x_squared + y_squared + z_squared
print(f"unit vector: {unit_vector}")
#this is needed to deal with the edge cases
test = quarternions[0]*quarternions[1] + quarternions[2]*quarternions[3]

##deal with gimbal lock
if test > (positive_gimbal_lock_cuttoff*unit_vector):
    yaw_angle = 2 * math.atan2(quarternions[0],quarternions[3]) ## heading
    pitch_angle = math.pi/2
    roll_angle = 0

##deal with gimbal lock
elif test < (negative_gimbal_lock_cuttoff*unit_vector):
    yaw_angle = -2 * math.atan2(quarternions[0],quarternions[3])
    pitch_angle = -1*(math.pi/2)
    roll_angle = 0

else: 
##sin_roll = (2*(xw+yz)) (bank)
    sin_roll = 2* (quarternions[0]*quarternions[3] + quarternions[1]*quarternions[2])
    ##cos_roll = (1-2*(x^2+z^2))
    cos_roll = 1 - 2*(x_squared + z_squared)

    ##sin_pitch = sqrt(1 + (2*(wy-xz))) (attitude)
    sin_pitch = math.sqrt(1 + 2*(quarternions[3]*quarternions[1] -quarternions[0]*quarternions[2]))
    ##sin_pitch = sqrt(1- (2*(wy-xz))_
    cos_pitch = math.sqrt(1 - 2*(quarternions[3]*quarternions[1] -quarternions[0]*quarternions[2]))

    ##sin_yaw = (2*(wy+xz)) (heading)
    sin_yaw = 2*(quarternions[3]*quarternions[1] - quarternions[0]*quarternions[2])
    ##cos_yaw = (1-2*(y^2+z^2))
    cos_yaw = 1 - 2*(y_squared + z_squared)

    print (f"roll components: {sin_roll}, {cos_roll}")
    print (f"pitch components: {sin_pitch}, {cos_pitch}")
    print (f"yaw components: {sin_pitch}, {cos_pitch}")


## we use atan2 to find the angle of each euler component instead of atan
## atan2 returns angles from -180 to 180 degrees  
    roll_angle = math.atan2(sin_roll, cos_roll)
    # pitch_angle = 2*(math.atan2(sin_pitch, cos_pitch)) - (math.pi/2)
    pitch_angle = math.asin(2*(quarternions[0]*quarternions[1]+quarternions[2]*quarternions[3]))
    yaw_angle = math.atan2(sin_yaw, cos_yaw)

print(f"roll angle: {math.degrees(roll_angle)}")
print(f"pitch angle: {math.degrees(pitch_angle)}")
print(f"yaw angle: {math.degrees(yaw_angle)}")



## potentially make a matrix using np.zeros((rows,colums)) of the same particle