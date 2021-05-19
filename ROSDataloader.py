import rospy
import numpy as np
from perception_msgs.msg import BoundingBoxTrajectoryArray

class TrajectoryCollector():
    # TODO: optimize this
    def __init__(self, collect = False) -> None:
        '''
        :params collect: keep all recieved uniq data if True
        '''
        self.trajectories = {}
        self.collect = collect
    
    def callback(self, msg):
        '''
        :params msg: ros msg BoundingBoxTrajectoryArray
        '''
        if not self.collect:
            self.trajectories.clear()

        for item in msg.trajectories:
            id = item.trajectory.boxes[0].label
            traj, ts = self.bboxes_to_coords(item.trajectory.boxes)
            if self.collect:
                # add all
                if id not in self.trajectories:
                    self.trajectories[id] = [[],[]]
                for pose in traj: 
                    self.trajectories[id][0].append(pose)    
                for timestamp in ts: 
                    self.trajectories[id][1].append(timestamp)
                # remove dublicates
                self.trajectories[id][0] = np.unique(self.trajectories[id][0], axis=0).tolist()
                self.trajectories[id][1] = np.unique(self.trajectories[id][1], axis=0).tolist()
            else:    
                # replace all
                self.trajectories[id] = [traj, ts]
        
    def bboxes_to_coords(self, bb):
        '''
        convert bounding box array to two lists - trajectory and timestamps
        :params bb: bounding box array ros msg(jsk_recognition_msgs/BoundingBox[])
        '''
        traj = []
        ts = []
        for b in bb:
            timestamp = b.header.stamp.to_time()*1000 #into ms
            coord = [b.pose.position.x, b.pose.position.y]
            ts.append(timestamp)
            traj.append(coord)
        return traj, ts

    def get_monoarray(self, timediff = None):
        '''
        :params timediff: minimun time between positions. all positions would be returned is timediff = None
        '''
        if len(self.trajectories)<1:
            return None
        out = []
        for id,val in self.trajectories.items():
            prev_ts = 0
            for i in range(len(val[0])):
                if timediff is not None:
                    ts = val[1][i]
                    diff = ts - prev_ts
                    if diff>=timediff:
                        prev_ts = ts
                        out.append([val[1][i], id, val[0][i][0], val[0][i][1]])
                else:
                    out.append([val[1][i], id, val[0][i][0], val[0][i][1]]) # id, ts, x, y
        return out


if __name__=="__main__":
    collector = TrajectoryCollector(collect=True)
    
    
    rospy.init_node('pedestrian_forecasting_dataloader', anonymous=True)
    rospy.Subscriber('/trajectories3d', BoundingBoxTrajectoryArray, collector.callback)
    # rate = rospy.Rate(1) # 10hz
    # while not rospy.is_shutdown():
    #     rate.sleep()
    #     data = collector.get_monoarray(timediff=None)
    
    rospy.sleep(30) # collect data over 40s
    data = collector.get_monoarray( timediff=350)
    with open("output.txt", "w") as outfile:
        for line in data:
            string = str(line)[1:-1].replace(", ", '\t')
            outfile.write('\t'+string+'\n')
    exit()
