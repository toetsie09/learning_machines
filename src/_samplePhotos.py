from robot_interface import RoboboEnv

#------------------------------
#   Used for taking photos of the hunted blocks
#   Run _camera_calibrator.py after taking good photos
#------------------------------

robobo = RoboboEnv(env_type='hardware', ip='192.168.2.11' , used_sensors='5')
robobo._env.set_phone_tilt(109,100)
robobo.takePhoto(times=10, count=10)


