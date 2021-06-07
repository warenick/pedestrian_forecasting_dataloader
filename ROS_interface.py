import rospy
import numpy as np
from perception_msgs.msg import BoundingBoxTrajectoryArray
from transformations import ChangeOrigin, Rotate
from utils import trajectory_orientation, transform_points, DataStructure
from dataloader import UnifiedInterface


class TrajectoryCollector():
    # TODO: optimize this
    # TODO: just do it later
    def __init__(self, collect=False, timedif = None) -> None:
        '''
        :params collect: keep all recieved uniq data if True
        :params timedif: set minimal difference in timestamp for data generation. all recieved positions would be returned if timediff = None
        '''
        self.timedif = timedif
        self.trajectories = {}
        self.collect = collect

    def _fancy_print_monoarray(self):
        sp = "\t"
        sn = "\t\n"
        ff = "{:10.3f}"
        output =  sp+"TrajectoryCollector state"+sn
        output += sp+"monoarray"+sn
        if self.monoarray is None: 
            output += sp+"no data"
            print(output)
        output += sp+"ts"+sp+sp+"id"+sp+sp+"x"+sp+sp+"y"+sp+sp+"diff_ts(calculated only for print)"+sn
        prev_ts = 0
        for line in self.monoarray:
            diff = line[0]-prev_ts
            prev_ts = line[0]
            output +=sp+ff.format(line[0])+sp+ff.format(line[1])+sp+ff.format(line[2])+sp+ff.format(line[3])+sp+ff.format(diff)+sn
        print(output)

    def __str__(self) -> str:
        sp = "\t"
        sn = "\t\n"
        ff = "{:10.3f}"
        # tsf = "{:10.4f}"
        output =  sp+"TrajectoryCollector state"+sn
        output += sp+"do collect all data="+str(self.collect)+sn
        output += sp+"id"+sp+"position"+sp+sp+sp+"timestamp"+sp+sp+sp+"diff time(calculated only for print)"+ sn
        if len(self.trajectories)<1:
            output +=sp+"no any data"
            return output
        prev_ts = 0
        for agent in self.trajectories.keys():
            if len(self.trajectories[agent][0])<1:
                output+= sp+str(agent)+sp+"no data"+sn
                continue
            for n in range(len(self.trajectories[agent][0])):
                diff = self.trajectories[agent][1][n]-prev_ts
                prev_ts = self.trajectories[agent][1][n]
                output+= sp+str(agent)+sp+ff.format(self.trajectories[agent][0][n][0])+sp+ff.format(self.trajectories[agent][0][n][1])+sp+ff.format(self.trajectories[agent][1][n])+sp+ff.format(diff)+sn
        return output

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
                    self.trajectories[id] = [[], []]
                for pose in traj:
                    self.trajectories[id][0].append(pose)
                for timestamp in ts:
                    self.trajectories[id][1].append(timestamp)
                # remove dublicates
                self.trajectories[id][0] = np.unique(
                    self.trajectories[id][0], axis=0).tolist()
                self.trajectories[id][1] = np.unique(
                    self.trajectories[id][1], axis=0).tolist()
            else:
                # replace all
                self.trajectories[id] = [traj, ts]
        self.batch = self.__generate_batch(self.timedif)
        self.monoarray = self.__generate_monoarray(self.timedif)

    def bboxes_to_coords(self, bb):
        '''
        convert bounding box array to two lists - trajectory and timestamps
        :params bb: bounding box array ros msg(jsk_recognition_msgs/BoundingBox[])
        '''
        traj = []
        ts = []
        for b in bb:
            timestamp = b.header.stamp.to_time()*1000  # into ms
            coord = [b.pose.position.x, b.pose.position.y]
            ts.append(timestamp)
            traj.append(coord)
        return traj, ts

    def get_monoarray(self):
        return self.monoarray

    def __generate_monoarray(self, timediff=None):
        '''
        :params timediff: minimun time between positions. all positions would be returned is timediff = None
        '''
        if len(self.trajectories) < 1:
            return None
        out = []
        for id, val in self.trajectories.items():
            prev_ts = 0
            for i in range(len(val[0])):
                if timediff is not None:
                    diff = val[1][i] - prev_ts
                    if diff >= timediff:
                        prev_ts = val[1][i]
                        out.append([val[1][i], id, val[0][i][0], val[0][i][1]])
                else:
                    # id, ts, x, y
                    out.append([val[1][i], id, val[0][i][0], val[0][i][1]])
        return out

    def __filter_timedif(self, trajectory, timedif=350):
        '''
        Filter pose array according difference between timestamps
        :params trajectory: (poses_array, ts_array)
        :params timedif: minimum difference time betwen positions

        :return out_poses: filtered array of poses
        :return out_ts: filtered array of timestamps with equal len to out_poses
        '''
        poses_array, ts_array = trajectory
        assert len(poses_array) == len(ts_array)
        out_poses = []
        out_ts = []
        prev_ts = 0
        for i in range(len(poses_array)):
            diff = ts_array[i] - prev_ts
            if diff >= timedif:
                prev_ts = ts_array[i]
                out_poses.append(poses_array[i])
                out_ts.append(ts_array[i])
        return out_poses, out_ts

    def __create_scene_chunk(self, agent, neighbors):
        '''
        :params agent: (list_of_pose, list_of_ts)
        :params neighbors: [(list_of_pose, list_of_ts),...]
        :return chunk: type of DataStructure class
        '''
        max_traj_len = 8
        data = DataStructure()
        data.pix_to_m = {"scale": np.eye(3)}
        # agent
        data.agent_pose = np.ones([max_traj_len, 2])
        data.agent_pose_av = np.zeros(max_traj_len)

        len_available = min(len(agent[0]), max_traj_len)

        # agent_pose
        data.agent_pose[:len_available] = agent[0][:len_available]
        data.agent_pose_av[:len_available] = 1
        # transform from pixels to meters
        data.agent_pose = transform_points(
            data.agent_pose, data.pix_to_m["scale"])

        data.target = np.zeros([12, 2])
        data.target_av = np.zeros(12)

        # neighbors
        num_neghbors = len(neighbors)
        data.neighb_poses = np.zeros([num_neghbors, max_traj_len, 2])
        data.neighb_poses_av = np.zeros([num_neghbors, max_traj_len])
        for agent in range(len(neighbors)):
            len_available = min(len(neighbors[agent][0]), max_traj_len)
            if len_available<1: continue # skip agent wothout data
            # agent_pose
            data.neighb_poses[agent][:len_available] = neighbors[agent][0][:len_available]
            data.neighb_poses_av[agent][:len_available] = 1
            # transform from pixels to meters
            data.neighb_poses[agent] = transform_points(
                data.neighb_poses[agent], data.pix_to_m["scale"])

        angle_deg = trajectory_orientation(data.agent_pose[0], data.agent_pose[1]) if len_available > 1 else 0
        co_operator = ChangeOrigin(new_origin=data.agent_pose[0], rotation=np.eye(2))
        r_operator = Rotate(angle=angle_deg, rot_center=data.agent_pose[0])
        to_localM_transform = co_operator.transformation_matrix @ r_operator.transformation_matrix
        data.agent_pose = transform_points(data.agent_pose, to_localM_transform)
        # TODO check scale multiplucation
        data.neighb_poses = transform_points(data.neighb_poses, to_localM_transform @ data.pix_to_m["scale"])
        data.raster_from_agent = np.linalg.inv(data.pix_to_m["scale"]) @ np.linalg.inv(to_localM_transform)
        # TODO check scale multiplucation
        data.raster_from_world = np.eye(3)
        
        data.world_from_agent = np.eye(3)
        data.agent_from_world = np.eye(3)
        return data

    def get_batch(self):
        return self.batch

    def __generate_batch(self, timediff=None):
        # self.trajectories: {id:[[poses],[ts]]}
        
        if len(self.trajectories) < 1: return None # check counts of agents
        list_of_chunks = []

        # filter around of time
        trajectories = dict.fromkeys(self.trajectories.keys()) if timediff is not None else self.trajectories
        if timediff is not None:
            for key in self.trajectories.keys():
                trajectories[key] = self.__filter_timedif(
                    self.trajectories[key], timediff)

        for agent in trajectories.keys():
            neighbors_id = list(trajectories.keys())
            neighbors_id.remove(agent)
            neighbors = [trajectories[id] for id in neighbors_id]
            if len(trajectories[agent][0])<1: continue # that agent dont have data 
            chunk = self.__create_scene_chunk(trajectories[agent], neighbors) 
            list_of_chunks.append(chunk)

        if len(list_of_chunks)<1: return None # no one in agents have data
        return UnifiedInterface(list_of_chunks)


class Visualiser():
    def __init__(self) -> None:
        pass
    def 
        

if __name__ == "__main__":
    # Convertor checking
    # list_of_traj = [[[1,0],[0,0],[1,2],[1,2],[1,2],[1,2],[1,2]], [[1,2],[1,2],[1,2],[1,2],[1,2],[1,2]]]
    # list_of_ts = [[1,2,3,4,5,6,7], [1,2,3,4,5,6]]
    # list_of_id = [1, 2]

    # scene = scene_from_data(list_of_traj, list_of_ts, list_of_id)
    # batch = get_batch_from_scene(scene)
    # print(batch.history_positions[0])

    # Collector checking
    rospy.init_node('pedestrian_forecasting_dataloader', anonymous=True)

    # Continues collect
    collector = TrajectoryCollector(collect=False, timedif=350)
    rospy.Subscriber('/trajectories3d',
                     BoundingBoxTrajectoryArray, collector.callback)
    rate = rospy.Rate(1)  # 10hz
    while not rospy.is_shutdown():
        rate.sleep()
        data  = collector.get_monoarray()
        batch = collector.get_batch()
        # print("created batch" if batch else "no data")
        print(collector)                    # print all collected data
        collector._fancy_print_monoarray()  # print filtered by timediff
        print(batch)                        # print filtered batch interpretation

    # Collect and save into file
    # collector = TrajectoryCollector(collect=True)
    # rospy.Subscriber('/trajectories3d', BoundingBoxTrajectoryArray, collector.callback)
    # rospy.sleep(30) # collect data over 40s
    # data = collector.get_monoarray( timediff=350)
    # # save collected data into file
    # with open("output.txt", "w") as outfile:
    #     for line in data:
    #         string = str(line)[1:-1].replace(", ", '\t')
    #         outfile.write('\t'+string+'\n')
    exit()
