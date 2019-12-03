#!/usr/bin/env python

import rospy
from asl_turtlebot.msg import DetectedObject
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String
import tf
import numpy as np
from numpy import linalg
from utils import wrapToPi
from planners import AStar, compute_smoothed_traj
from grids import StochOccupancyGrid2D
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum
from visualization_msgs.msg import Marker
import time

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    STOP=4
    CROSS=5

class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """
    def __init__(self):
        rospy.init_node('turtlebot_navigator', anonymous=True)
        self.mode = Mode.IDLE
        self.xlist=[]
        self.ylist=[]
        self.tlist=[]
        self.object_dic={}

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None

        self.th_init = 0.0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0,0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution =  0.1
        self.plan_horizon = 15

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.,0.]
        
        # Robot limits
        self.v_max=0.1
        self.om_max=0.2

        self.v_des = 0.12   # desired cruising velocity
        self.theta_start_thresh = 0.05   # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = 0.2     # threshold to be far enough into the plan to recompute it

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.2
        self.at_thresh = 0.02
        self.at_thresh_theta = 0.05

        # trajectory smoothing
        self.spline_alpha = 0.15
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 0.5
        self.kpy = 0.5
        self.kdx = 1.5
        self.kdy = 1.5

        # heading controller parameters
        self.kp_th = 2.

        self.traj_controller = TrajectoryTracker(self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max)
        self.pose_controller = PoseController(0., 0., 0., self.v_max, self.om_max)
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.nav_smoothed_path_pub = rospy.Publisher('/cmd_smoothed_path', Path, queue_size=10)
        self.nav_smoothed_path_rej_pub = rospy.Publisher('/cmd_smoothed_path_rejected', Path, queue_size=10)
        self.nav_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.vis_pub = rospy.Publisher('marker_topic', Marker, queue_size=10)

        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)

        self.stop_min_dist=1.0
        self.stop_time=3.0
        self.crossing_time=10.0
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/map_metadata', MapMetaData, self.map_md_callback)
        rospy.Subscriber('/cmd_nav', Pose2D, self.cmd_nav_callback)
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
        rospy.Subscriber('/detector/bird', DetectedObject, self.bird_callback)
        rospy.Subscriber('/detector/cake', DetectedObject, self.cake_callback)
        rospy.Subscriber('/detector/pizza', DetectedObject, self.pizza_callback)
        rospy.Subscriber('/detector/bottle', DetectedObject, self.bottle_callback)
        self.initPos=False
        self.auto=False
        self.bird=0
        self.cake=1
        self.pizza=2
        self.bottle=3

        print "finished init"
    
    def startPlan(self):
        for i in range(4):
            if i==self.bird:
                print i
                print "bird"
                if 'bird' in self.object_dic:
                    x,y,t=self.object_dic['bird']
                    self.xlist.append(x)
                    self.ylist.append(y)
                    self.tlist.append(t)
            elif i==self.cake:  
                print i
                print "cake"
                if 'cake' in self.object_dic:
                    x,y,t=self.object_dic['cake']
                    self.xlist.append(x)
                    self.ylist.append(y)
                    self.tlist.append(t)
            elif i==self.pizza:
                print i
                print "pizza"
                if 'pizza' in self.object_dic:
                    x,y,t=self.object_dic['pizza']
                    self.xlist.append(x)
                    self.ylist.append(y)
                    self.tlist.append(t)
            elif i==self.bottle:
                print i
                print "bottle"
                if 'bottle' in self.object_dic:
                    x,y,t=self.object_dic['bottle']
                    self.xlist.append(x)
                    self.ylist.append(y)
                    self.tlist.append(t)
        print 4
        print "Destination"
        x,y,t=self.object_dic['start']
        self.xlist.append(x)
        self.ylist.append(y)
        self.tlist.append(t)

    def show(self,idx,x,y,r,g,b,t):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()
        marker.id = idx
        marker.type = t # sphere
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        self.vis_pub.publish(marker)        

    def bird_callback(self,msg):
        self.object_dic['bird']=(self.x,self.y,self.theta)
        print 'bird'
        print self.object_dic['bird']
        self.show(0,self.x,self.y,1.0,0.0,0.0,2)

    def cake_callback(self,msg):
        self.object_dic['cake']=(self.x,self.y,self.theta)
        print 'cake'
        print self.object_dic['cake']
        self.show(1,self.x,self.y,0.0,1.0,0.0,2)

    def pizza_callback(self,msg):
        self.object_dic['pizza']=(self.x,self.y,self.theta)
        print 'pizza'
        print self.object_dic['pizza']
        self.show(2,self.x,self.y,0.0,0.0,1.0,2)

    def bottle_callback(self,msg):
        self.object_dic['bottle']=(self.x,self.y,self.theta)
        print 'bottle'
        print self.object_dic['bottle']
        self.show(2,self.x,self.y,0.0,1.0,1.0,2)

    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        # print "Stop Sign Destected"
        dist = msg.distance
        # if self.mode==Mode.TRACK:
        # if close enough and in nav mode, stop
        if dist > 0 and dist < self.stop_min_dist and self.mode == Mode.TRACK:
            print "Stop Sign Countdown"
            self.init_stop_sign()

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """

        self.stop_sign_start = rospy.get_rostime()
        self.mode = Mode.STOP
        # print "initial stop sign"
        # print self.stop_sign_start

    def has_stopped(self):
        """ checks if stop sign maneuver is over """

        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.stop_time)

    def init_crossing(self):
        """ initiates an intersection crossing maneuver """

        self.cross_start = rospy.get_rostime()
        self.mode = Mode.CROSS
        # print "Start crossing"
        # print self.cross_start

    def has_crossed(self):
        """ checks if crossing maneuver is over """

        return self.mode == Mode.CROSS and \
               rospy.get_rostime() - self.cross_start > rospy.Duration.from_sec(self.crossing_time)


    def dyn_cfg_callback(self, config, level):
        # rospy.loginfo("Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3},vm:{vm},sa:{sa},td:{td}".format(**config))
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        self.v_max=config["vm"]
        self.spline_alpha = config["sa"]
        self.traj_dt = config["td"]
        self.bird=config["bird"]
        self.cake=config["cake"]
        self.pizza=config["pizza"]
        self.bottle=config["bottle"]
        self.auto=config["auto"]
        return config

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        if data.x != self.x_g or data.y != self.y_g or data.theta != self.theta_g:
            self.x_g=data.x
            self.y_g=data.y
            self.theta_g=data.theta
            self.replan()
        # print(self.x_g,self.y_g,self.theta_g)

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x,msg.origin.position.y)

    def map_callback(self,msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if self.map_width>0 and self.map_height>0 and len(self.map_probs)>0:
            self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  self.map_width,
                                                  self.map_height,
                                                  self.map_origin[0],
                                                  self.map_origin[1],
                                                  8,
                                                  self.map_probs)
            if self.x_g is not None:
                # if we have a goal to plan to, replan
                # rospy.loginfo("replanning because of new map")
                self.replan() # new map, need to replan

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.near_thresh

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        return (linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.near_thresh and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta)

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh)
        
    def close_to_plan_start(self):
        return (abs(self.x - self.plan_start[0]) < self.start_pos_thresh and abs(self.y - self.plan_start[1]) < self.start_pos_thresh)

    def snap_to_grid(self, x):
        return (self.plan_resolution*round(x[0]/self.plan_resolution), self.plan_resolution*round(x[1]/self.plan_resolution))

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i,0]
            pose_st.pose.position.y = traj[i,1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.TRACK or self.mode==Mode.CROSS:
            V, om = self.traj_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(self.x, self.y, self.theta, t)
        else:
            V = 0.
            om = 0.

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime()-self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo("Navigator: replanning canceled, waiting for occupancy map.")
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem = AStar(state_min,state_max,x_init,x_goal,self.occupancy,self.plan_resolution)

        rospy.loginfo("Navigator: computing navigation plan")
        success =  problem.solve()
        if not success:
            rospy.loginfo("Planning failed")
            return
        rospy.loginfo("Planning Succeeded")

        planned_path = problem.path
        

        # Check whether path is too short
        if len(planned_path) < 4:
            rospy.loginfo("Path too short to track")
            self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        traj_new, t_new = compute_smoothed_traj(planned_path, self.v_des, self.spline_alpha, self.traj_dt)

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = self.current_plan_duration - self.get_current_plan_time()

            # Estimate duration of new trajectory
            th_init_new = traj_new[0,2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err/self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                # rospy.loginfo("New plan rejected (longer duration than current plan)")
                self.publish_smoothed_path(traj_new, self.nav_smoothed_path_rej_pub)
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0,2]
        self.heading_controller.load_goal(self.th_init)

        if self.mode==Mode.STOP or self.mode==Mode.CROSS:
            return

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return
            
        rospy.loginfo("Ready to track")
        self.switch_mode(Mode.TRACK)

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
                if self.initPos==False:
                    self.object_dic['start']=(self.x,self.y,self.theta)
                    print 'start position'
                    print self.object_dic['start']
                    self.initPos=True
                    self.show(3,self.x,self.y,1.0,1.0,0.0,1)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                self.current_plan = []
                # rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print e
                pass
            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                if self.auto:
                    print "Delivery start"
                    self.startPlan()
                    self.auto=False
                if self.xlist:
                    print "Read next goal"
                    time.sleep(3)
                    self.x_g=self.xlist.pop(0)
                    self.y_g=self.ylist.pop(0)
                    self.theta_g=self.tlist.pop(0)
                    self.replan()
                else:
                    pass
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    # rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (rospy.get_rostime() - self.current_plan_start_time).to_sec() > self.current_plan_duration:
                    # rospy.loginfo("replanning because out of time")
                    self.replan() # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.PARK:
                if self.at_goal():
                    # forget about goal:
                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    self.switch_mode(Mode.IDLE)
            elif self.mode==Mode.STOP:
                if self.has_stopped():
                    print rospy.get_rostime()
                    self.init_crossing()
            elif self.mode == Mode.CROSS:
                # Crossing an intersection
                if self.has_crossed():
                    print rospy.get_rostime()
                    self.mode=Mode.TRACK

            self.publish_control()
            rate.sleep()

if __name__ == '__main__':    
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    nav.run()
