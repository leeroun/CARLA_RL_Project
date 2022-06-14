from keras.layers import  Dense, Concatenate, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Sequential, Input, Model
from keras.callbacks import TensorBoard

from collections import deque

import tensorflow as tf
import random
import time
import numpy as np

import carla
import cv2
import math

MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5

MODEL_INPUTS = 5
ACTION_NUMBER = 4

DISCOUNT = 0.99

SECONDS_PER_EPISODE = 15

REPLAY_MEMORY_SIZE = 5_000
MODEL_NAME = "DNN_V2_l2_A4"
SHOW_CAM = False


class ModifiedTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0
        self.graph = tf.compat.v1.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        input_shape = Input(shape=(16,))
        # left_dis (1) right_dis (1) delta angle (1) left lane type (4) right lane type (4)
        # Accel (1) Gyro_z (1)
        # Left Dis (1) Front Dis (1) Right Dis (1)

        d1out = Dense(32, activation='relu')(input_shape)
        d2out = Dense(16, activation='relu')(d1out)
        dout = Dense(ACTION_NUMBER, activation='linear')(d2out)

        model = Model(inputs=input_shape, outputs=dout)

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        model.summary()
        print(model.input_shape)
        return model

    def update_replay_memory(self, transition):
        transition = np.array(transition)
        current_transition = transition[0]
        new_transition = transition[3]

        cur_list = [current_transition[0], current_transition[1], current_transition[2],
                current_transition[3][0], current_transition[3][1],  current_transition[3][2],  current_transition[3][3],
                current_transition[4][0], current_transition[4][1],  current_transition[4][2],  current_transition[4][3],
                current_transition[5], current_transition[6], current_transition[7][0], current_transition[7][1], current_transition[7][2]]
        cur_list = np.expand_dims(cur_list, axis=0)

        transition[0] = [cur_list]

        new_list = [new_transition[0], new_transition[1], new_transition[2],
                new_transition[3][0], new_transition[3][1],  new_transition[3][2],  new_transition[3][3],
                new_transition[4][0], new_transition[4][1],  new_transition[4][2],  new_transition[4][3],
                new_transition[5], new_transition[6], new_transition[7][0], new_transition[7][1], new_transition[7][2]]
        new_list = np.expand_dims(new_list, axis=0)

        transition[3] = [new_list]

        self.replay_memory.append(transition)
        # print(f'replay memory length : {len(self.replay_memory)} / {REPLAY_MEMORY_SIZE}')

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = [transition[0] for transition in minibatch]

        current_qs_list = []
        with self.graph.as_default():
            for i in range(MINIBATCH_SIZE):
                current_qs_list.append(self.model.predict(current_states[i], PREDICTION_BATCH_SIZE)[0])

        new_current_states = [transition[3] for transition in minibatch]

        future_qs_list = []
        with self.graph.as_default():
            for i in range(MINIBATCH_SIZE):
                future_qs_list.append(self.target_model.predict(new_current_states[i], PREDICTION_BATCH_SIZE)[0])

        X = []
        Y = []

        for index, (current_states, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_states)
            Y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        with self.graph.as_default():
            for i in range(MINIBATCH_SIZE):
                _Y = np.expand_dims(np.array(Y[i]), axis=0)
                self.model.fit(X[i], _Y, batch_size=1, verbose=0, shuffle=False)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        # [0]left_dis (1) [1]right_dis (1) [2]delta angle (1) [3]left lane type (4) [4]right lane type (4)
        # [5]Accel (1) [6]Gyro_z (1)
        # [7]Dis (3) [Left Front Right]
        list = [state[0], state[1], state[2],
                state[3][0], state[3][1],  state[3][2],  state[3][3],
                state[4][0], state[4][1],  state[4][2],  state[4][3],
                state[5], state[6], state[7][0], state[7][1], state[7][2]]
        qs_list = np.expand_dims(list, axis=0)

        return self.model.predict([qs_list])[0]

    def train_in_loop(self):
        # left_dis (1) right_dis (1) delta angle (1) left lane type (4) right lane type (4)
        # Accel (1) Gyro (3)
        # Dis (3) [Left Front Right]

        X_left_dis = np.random.uniform(size=1).astype(np.float32)
        X_right_dis = np.random.uniform(size=1).astype(np.float32)
        X_delta_angle = np.random.uniform(size=1).astype(np.float32)
        X_left_type = np.eye(4)[np.random.randint(0, 4)].astype(np.float32)
        X_right_type = np.eye(4)[np.random.randint(0, 4)].astype(np.float32)
        X_accel = np.random.uniform(size=1).astype(np.float32)
        X_gyro = np.random.uniform(size=1).astype(np.float32)
        X_dis = np.random.uniform(size=3).astype(np.float32)

        X = np.concatenate((X_left_dis, X_right_dis, X_delta_angle, X_left_type, X_right_type, X_accel, X_gyro, X_dis))
        X = np.expand_dims(X, axis=0)

        # X = np.array([X_image_f, X_image_l, X_image_r, X_lane])
        y = np.random.uniform(size=(1, ACTION_NUMBER)).astype(np.float32)

        with self.graph.as_default():
            self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


class CarEnv:

    def __init__(self, result=False):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.result = result

        self.vehicle = None

        self.handle_value = 0.0
        self.throttle_value = 0.0

        self.episode_start = 0

        self.left_distance = 0
        self.right_distance = 0
        self.front_distance = 0

        self.accel = 0
        self.gyro_z = 0

        self.spawn_points = self.world.get_map().get_spawn_points()
        self.actor_start_point = carla.Transform()

        self.vehicle_blueprints = self.blueprint_library.filter('*vehicle*')

    def reset(self):

        self.collision_hist = []
        self.actor_list = []
        self.vehicle_list = []

        while True:
            vehicle_transform = random.choice(self.spawn_points)
            self.actor_start_point = vehicle_transform
            self.vehicle = self.world.try_spawn_actor(self.model_3, vehicle_transform)

            spec_loc = carla.Location(-7, 0, 6)
            vehicle_transform.transform(spec_loc)
            spectator_transform = carla.Transform(carla.Location(spec_loc.x, spec_loc.y, spec_loc.z),
                                                  carla.Rotation(pitch=vehicle_transform.rotation.pitch-25, yaw=vehicle_transform.rotation.yaw, roll=vehicle_transform.rotation.roll))

            self.world.get_spectator().set_transform(spectator_transform)
            if self.vehicle is not None:
                self.actor_list.append(self.vehicle)
                break

        for i in range(0, 150):
            tmp_vehicle = self.world.try_spawn_actor(random.choice(self.vehicle_blueprints), random.choice(self.spawn_points))
            if tmp_vehicle is not None:
                tmp_vehicle.set_autopilot(True)
                self.vehicle_list.append(tmp_vehicle)

        col_sensor_bp = self.blueprint_library.find("sensor.other.collision")
        col_sensor = self.world.spawn_actor(col_sensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(col_sensor)
        col_sensor.listen(lambda event: self.collision_data(event))

        radar_sensor_bp = self.blueprint_library.find('sensor.other.radar')
        radar_sensor_bp.set_attribute('horizontal_fov', '110')
        radar_sensor_bp.set_attribute('vertical_fov', '15')
        radar_sensor_bp.set_attribute('range', '8')
        rad_loc = carla.Location(x=2.0, z=1.0)
        rad_rot = carla.Rotation(pitch=5)
        radar_sensor = self.world.spawn_actor(radar_sensor_bp, carla.Transform(rad_loc, rad_rot),
                                              attach_to=self.vehicle)
        radar_sensor.listen(lambda data: self.radar_callback(data))
        self.actor_list.append(radar_sensor)

        IMU_sensor_bp = self.blueprint_library.find('sensor.other.imu')
        IMU_sensor = self.world.spawn_actor(IMU_sensor_bp, carla.Transform(), attach_to=self.vehicle)
        IMU_sensor.listen(lambda data: self.IMU_callback(data))
        self.actor_list.append(IMU_sensor)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

        return self.get_state()

    def get_state(self):
        lane_left_dis = 0
        lane_left_type = [1, 0, 0, 0]
        lane_right_dis = 0
        lane_right_type = [1, 0, 0, 0]
        sensing_distance = [0, 0, 0]

        waypoint = self.client.get_world().get_map().get_waypoint(self.vehicle.get_location(), project_to_road=True)
        vehicle_location = self.vehicle.get_location()

        vehicle_forward = self.vehicle.get_transform().get_forward_vector()
        waypoint_forward = waypoint.transform.get_forward_vector()
        deltaAngle = math.acos(
            (
                    vehicle_forward.x * waypoint_forward.x + vehicle_forward.y * waypoint_forward.y + vehicle_forward.z * waypoint_forward.z) /
            math.sqrt(
                vehicle_forward.x * vehicle_forward.x + vehicle_forward.y * vehicle_forward.y + vehicle_forward.z * vehicle_forward.z) /
            math.sqrt(
                waypoint_forward.x * waypoint_forward.x + waypoint_forward.y * waypoint_forward.y + waypoint_forward.z * waypoint_forward.z)
        )
        deltaAngle *= math.degrees(deltaAngle)
        lane_delta_angle = deltaAngle

        left_lane_point = waypoint.get_left_lane()
        right_lane_point = waypoint.get_right_lane()
        if left_lane_point is not None and right_lane_point is not None:
            left_lane_location = left_lane_point.transform.location
            right_lane_location = right_lane_point.transform.location

            left_distance = math.sqrt(math.pow((vehicle_location.x - left_lane_location.x), 2)
                                      + math.pow((vehicle_location.y - left_lane_location.y), 2)
                                      + math.pow((vehicle_location.z - left_lane_location.z), 2))
            right_distance = math.sqrt(math.pow((vehicle_location.x - right_lane_location.x), 2)
                                       + math.pow((vehicle_location.y - right_lane_location.y), 2)
                                       + math.pow((vehicle_location.z - right_lane_location.z), 2))
            lane_left_dis = left_distance
            lane_right_dis = right_distance
            lane_left_type = self.lane_type_to_onehot_Vector(waypoint.left_lane_marking.type)
            lane_right_type = self.lane_type_to_onehot_Vector(waypoint.left_lane_marking.type)

        # normalization
        lane_width = waypoint.lane_width * 2

        lane_left_dis /= lane_width
        lane_right_dis /= lane_width

        lane_left_dis = min(1.0, lane_left_dis)
        lane_right_dis = min(1.0, lane_right_dis)

        lane_delta_angle /= 180

        imu_accel = min(self.accel, 50)/50

        imu_gyro = (math.degrees(self.gyro_z)+90)/180

        sensing_distance[0] = self.left_distance / 10
        sensing_distance[1] = self.front_distance / 10
        sensing_distance[2] = self.right_distance / 10

        # print(f'Lane Left Dist {lane_left_dis} | Lane Right Dist {lane_right_dis} | Lane Delta Angle {lane_delta_angle}')
        # print(f'Lane Left Type {lane_left_type} | Lane Right Type {lane_right_type}')
        # print(f'Accel {imu_accel} | Gyro {imu_gyro}')
        # print(f'Sensing Distance {sensing_distance}')

        # left_dis (1) right_dis (1) delta angle (1) left lane type (4) right lane type (4)
        # Accel (1) Gyro_z (1)
        # Left Dis (1) Front Dis (1) Right Dis (1)
        return [lane_left_dis, lane_right_dis, lane_delta_angle, lane_left_type, lane_right_type, imu_accel, imu_gyro, sensing_distance]

    def collision_data(self, event):
        self.collision_hist.append(event)

    def radar_callback(self, data):
        if not self.vehicle.is_alive:
            return
        current_rot = data.transform.rotation
        vehicle_location = self.vehicle.get_transform().location
        vehicle_forward = self.vehicle.get_transform().get_forward_vector()

        left_list = []
        right_list = []
        front_list = []

        for detect in data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            forward = carla.Vector3D(x=detect.depth - 0.25)

            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(forward)

            detect_transform = data.transform.location + forward

            detect_vector = carla.Vector3D(detect_transform.x - vehicle_location.x,
                                           detect_transform.y - vehicle_location.y,
                                           detect_transform.z - vehicle_location.z)

            distance = math.sqrt(math.pow(detect_vector.x, 2)
                                 + math.pow(detect_vector.y, 2)
                                 + math.pow(detect_vector.z, 2))

            deltaAngle = math.acos(
                (vehicle_forward.x * detect_vector.x + vehicle_forward.y * detect_vector.y + vehicle_forward.z * detect_vector.z) /
                math.sqrt(vehicle_forward.x * vehicle_forward.x + vehicle_forward.y * vehicle_forward.y + vehicle_forward.z * vehicle_forward.z) /
                math.sqrt(detect_vector.x * detect_vector.x + detect_vector.y * detect_vector.y + detect_vector.z * detect_vector.z)
            )

            if np.cross([vehicle_forward.x, vehicle_forward.y, vehicle_forward.z],
                        [detect_vector.x, detect_vector.y, detect_vector.z])[2] > 0:
                sign = 1
            else:
                sign = 0

            deltaAngle *= math.degrees(deltaAngle)

            if deltaAngle > 10:
                if sign > 0:
                    right_list.append(distance)
                else:
                    left_list.append(distance)
            else:
                front_list.append(distance)

        if len(left_list) != 0:
            self.left_distance = np.mean(left_list)
        if len(right_list) != 0:
            self.right_distance = np.mean(right_list)
        if len(front_list) != 0:
            self.front_distance = np.mean(front_list)

    def IMU_callback(self, data):
        accel_value = math.sqrt(data.accelerometer.x * data.accelerometer.x + data.accelerometer.y * data.accelerometer.y + data.accelerometer.z *data.accelerometer.z)
        self.accel = accel_value
        self.gyro_z = data.gyroscope.z

    def lane_type_to_onehot_Vector(self, lane_type):
        if lane_type is carla.LaneMarkingType.Broken:
            return [1.0, 0.0, 0.0, 0.0]
        elif lane_type is carla.LaneMarkingType.SolidSolid:
            return [0.0, 1.0, 0.0, 0.0]
        elif lane_type is carla.LaneMarkingType.Solid:
            return [0.0, 0.0, 1.0, 0.0]
        else:
            return [0.0, 0.0, 0.0, 1.0]

    def step(self, action):
        brake_value = 0
        if action == 0:
            self.throttle_value = 0.5
        elif action == 1:
            self.handle_value = -1
        elif action == 2:
            self.handle_value = 1
        elif action == 3:
            brake_value = 1

        # print(f'throttle: {self.throttle_value} steer" {self.handle_value}')
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=self.throttle_value, steer=self.handle_value, brake=brake_value))

        if self.result:
            return self.get_state(), 0, False, None

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        reward = 0

        # 충돌 reward
        if len(self.collision_hist) != 0:
            done = True
            reward -= 200
        else:

            waypoint_location = self.client.get_world().get_map().get_waypoint(self.vehicle.get_location(),
                                                                               project_to_road=True).transform.location
            vehicle_location = self.vehicle.get_location()
            distance = math.sqrt(math.pow((vehicle_location.x - waypoint_location.x), 2)
                                 + math.pow((vehicle_location.y - waypoint_location.y), 2)
                                 + math.pow((vehicle_location.z - waypoint_location.z), 2))

            distance_value = int(distance / 4)
            reward -= distance_value

            # 차량 속도에 따른 reward
            # velocity_value = int(kmh * math.sqrt(kmh) / 50 - (1 / 2))  # 9kmh 전까진 -1, 50kmh에서 대략 6정도
            # velocity_value = int(5 * math.log(kmh+1) - 5)  # 10kmh 전까진 -1, 50kmh에서 대략 3정도
            velocity_value = int(kmh / 10 - 3)

            reward += velocity_value

            # 이동 거리에 따른 reward
            mileage = math.sqrt(math.pow((vehicle_location.x - self.actor_start_point.location.x), 2)
                                + math.pow((vehicle_location.y - self.actor_start_point.location.y), 2)
                                + math.pow((vehicle_location.z - self.actor_start_point.location.z), 2))
            mileage_value = int(mileage * math.sqrt(mileage) / 40 - 1)
            reward += mileage_value

            # print(f'velocity {velocity_value} | mileage {mileage_value} | distance -{distance_value}')

            done = False

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.get_state(), reward, done, None

    def destroy_actors(self):
        # print(f'destroy actors({len(self.actor_list)})')
        # print(f'destroy vehicles({len(self.vehicle_list)})')
        # self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_list])

        for actor in self.actor_list:
            actor.destroy()

        self.actor_list.clear()
        self.vehicle_list.clear()
