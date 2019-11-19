#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped

class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """
    def __init__(self):
        # goal state
        self.x_g = 0.0
        self.y_g = 0.0
        self.theta_g = 0.0

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
	self.x_g = data.x
	self.y_g = data.y
	self.theta_g = data.theta

    def publisher(self):
        vis_pub = rospy.Publisher('marker_topic', Marker, queue_size=10)
        rospy.init_node('marker_node', anonymous=True)
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            rospy.Subscriber('/cmd_nav', Pose2D, self.cmd_nav_callback)
            marker = Marker()

            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time()

        # IMPORTANT: If you're creating multiple markers, 
        #            each need to have a separate marker ID.
            marker.id = 0

            marker.type = 2 # sphere

            marker.pose.position.x = self.x_g
            marker.pose.position.y = self.y_g
            marker.pose.position.z = self.theta_g

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
    
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.color.a = 1.0 # Don't forget to set the alpha!
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        
            vis_pub.publish(marker)
            print('Published marker!')
        
            rate.sleep()


if __name__ == '__main__':
    try:
	nav=Navigator()
        nav.publisher()
    except rospy.ROSInterruptException:
        pass
