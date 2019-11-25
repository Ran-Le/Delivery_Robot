#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose2D
import time

xlist=[3.07, 3.41, 3.13]
ylist=[1.73, 2.72, 1.16]
tlist=[1.83, -3.08, -1.12]
rospy.init_node('destination_publisher', anonymous=True)
nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
time.sleep(5)
while xlist:
    pose_g_msg = Pose2D()
    pose_g_msg.x = xlist.pop()
    pose_g_msg.y = ylist.pop()
    pose_g_msg.theta = tlist.pop()
    nav_goal_publisher.publish(pose_g_msg)
    time.sleep(20)

