import numpy as np
from utils import wrapToPi
import rospy
from std_msgs.msg import Float32

# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1


class PoseController:
    """ Pose stabilization controller """

    def __init__(self, k1, k2, k3, V_max=0.5, om_max=1):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max

	self.pub1 = rospy.Publisher('/controller/alpha', Float32, queue_size=10)
	self.pub2 = rospy.Publisher('/controller/delta', Float32, queue_size=10)
	self.pub3 = rospy.Publisher('/controller/rho', Float32, queue_size=10)

    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs: 
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########
        temp = (np.arctan2(self.y_g - y, self.x_g - x))
        alpha = wrapToPi(temp - th)
        delta = wrapToPi(temp - self.th_g)
        rho = np.sqrt((self.x_g - x)**2 + (self.y_g - y)**2)
        V = self.k1 * rho * np.cos(alpha)
        om = self.k2 * alpha + self.k1 * \
            (np.sinc(alpha / np.pi) * np.cos(alpha)) * (alpha + self.k3 * delta)

        my_message1 = Float32()
	my_message1.data=alpha
        self.pub1.publish(my_message1)
        my_message2 = Float32()
	my_message2.data=delta
        self.pub2.publish(my_message2)
        my_message3 = Float32()
	my_message3.data=rho
        self.pub3.publish(my_message3)

        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om
