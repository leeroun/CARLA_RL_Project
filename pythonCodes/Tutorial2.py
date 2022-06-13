import carla
import random
import time
import numpy as np
import cv2
from queue import Queue
from queue import Empty

import math

actor_list = []
vehicle_list = []
walker_list = []
walker_controller_list = []
IM_WIDTH = 640
IM_HEIGHT = 480

rgb_img = np.empty((IM_HEIGHT, IM_WIDTH, 3))
dep_img = np.empty((IM_HEIGHT, IM_WIDTH, 3))
f_seg_img = np.empty((IM_HEIGHT, IM_WIDTH, 3))
l_seg_img = np.empty((IM_HEIGHT, IM_WIDTH, 3))
r_seg_img = np.empty((IM_HEIGHT, IM_WIDTH, 3))

client = carla.Client('localhost', 2000)
client.load_world('Town10HD')
world = client.get_world()


def process_img(image):
    global rgb_img
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    rgb_img = i2[:, :, :3]

    render_image()
    return rgb_img / 255.0


def process_depth_img(image):
    global dep_img
    image.convert(carla.ColorConverter.LogarithmicDepth)
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    dep_img = i2[:, :, :3]

    render_image()
    return dep_img / 255.0


def process_segmentation_img(image, direction):
    global f_seg_img
    global l_seg_img
    global r_seg_img

    image.convert(carla.ColorConverter.CityScapesPalette)
    i = np.array(image.raw_data)

    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]

    if direction == 0:
        f_seg_img = i3
    if direction == 1:
        l_seg_img = i3
    if direction == 2:
        r_seg_img = i3

    render_image()
    return i3 / 255.0


def render_image():
    img = np.hstack((l_seg_img, f_seg_img, r_seg_img))
    cv2.imshow("sensors", img)
    # cv2.imshow("sensors", rgb_img)
    cv2.waitKey(1)


def on_invasion(event):
    lane_types = set(x.type for x in event.crossed_lane_markings)
    # print(lane_types)
    # for x in lane_types:
    #     print(f'Crossed line {str(x)}')


def clamp(min_v, max_v, value):
    return max(min_v, min(value, max_v))


def IMU_callback(IMU_data):
    accel = math.sqrt(IMU_data.accelerometer.x * IMU_data.accelerometer.x + IMU_data.accelerometer.y * IMU_data.accelerometer.y + IMU_data.accelerometer.z *IMU_data.accelerometer.z)
    print(f'accelerometer: {accel} | gyroscope {math.degrees(IMU_data.gyroscope.x)}, {math.degrees(IMU_data.gyroscope.y)}, {math.degrees(IMU_data.gyroscope.z)}' )


def radar_callback(radar_data, vehicle):
    current_rot = radar_data.transform.rotation
    vehicle_location = vehicle.get_transform().location
    vehicle_forward = vehicle.get_transform().get_forward_vector()

    left_distance=0
    right_distance=0
    front_distance=0

    left_list = []
    right_list = []
    front_list = []
    for detect in radar_data:

        azi = math.degrees(detect.azimuth)
        alt = math.degrees(detect.altitude)
        forward = carla.Vector3D(x=detect.depth - 0.25)

        carla.Transform(
            carla.Location(),
            carla.Rotation(
                pitch=current_rot.pitch + alt,
                yaw=current_rot.yaw + azi,
                roll=current_rot.roll)).transform(forward)

        detect_transform = radar_data.transform.location + forward

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

        color = carla.Color(255, 255, 255)

        deltaAngle *= math.degrees(deltaAngle)

        if deltaAngle > 10:
            if sign > 0:
                color = carla.Color(255, 0, 0)
                right_list.append(distance)
            else:
                color = carla.Color(0, 0, 255)
                left_list.append(distance)
        else:
            front_list.append(distance)

        world.debug.draw_point(detect_transform,
                               size=0.075,
                               life_time=0.06,
                               persistent_lines=False,
                               color=color)

    if len(left_list) != 0:
        left_distance = np.mean(left_list)
    if len(right_list) != 0:
        right_distance = np.mean(right_list)
    if len(front_list) != 0:
        front_distance = np.mean(front_list)

    print(f'left {left_distance} | right {right_distance} | front {front_distance}')

def main():
    # environment 연결

    # world
    bp_library = world.get_blueprint_library()
    bp = bp_library.filter("model3")[0]

    spawn_points = [carla.Transform(carla.Location(-30, -60.5, 0.6), carla.Rotation(0, 0, 0)),
                    carla.Transform(carla.Location(10, -57.5, 0.6), carla.Rotation(0, 0, 0)),
                    carla.Transform(carla.Location(10, -60.5, 0.6), carla.Rotation(0, 0, 0)),
                    carla.Transform(carla.Location(10, -64.7, 0.6), carla.Rotation(0, 0, 0))]
    # carla.Transform(carla.Location(18, -57.5, 0.6), carla.Rotation(0, 0, 0)),
    # carla.Transform(carla.Location(45, -57.5, 0.6), carla.Rotation(0, 0, 0)),
    # carla.Transform(carla.Location(-20, -60.5, 0.6), carla.Rotation(0, 0, 0)),
    # carla.Transform(carla.Location(11, -60.5, 0.6), carla.Rotation(0, 0, 0)),
    # carla.Transform(carla.Location(29, -60.5, 0.6), carla.Rotation(0, 0, 0)),
    # carla.Transform(carla.Location(-14, -68.3, 0.6), carla.Rotation(0, 180, 0)),
    # carla.Transform(carla.Location(0, -68.3, 0.6), carla.Rotation(0, 180, 0)),
    # carla.Transform(carla.Location(10, -68.3, 0.6), carla.Rotation(0, 180, 0)),
    # carla.Transform(carla.Location(29, -68.3, 0.6), carla.Rotation(0, 180, 0)),
    # carla.Transform(carla.Location(45, -68.3, 0.6), carla.Rotation(0, 180, 0)),
    # carla.Transform(carla.Location(55, -68.3, 0.6), carla.Rotation(0, 180, 0)),
    # carla.Transform(carla.Location(-10, -64.7, 0.6), carla.Rotation(0, 180, 0)),
    # carla.Transform(carla.Location(3, -64.7, 0.6), carla.Rotation(0, 180, 0)),
    # carla.Transform(carla.Location(21, -64.7, 0.6), carla.Rotation(0, 180, 0)),
    # carla.Transform(carla.Location(30, -64.7, 0.6), carla.Rotation(0, 180, 0)),
    # carla.Transform(carla.Location(38, -64.7, 0.6), carla.Rotation(0, 180, 0)),
    # carla.Transform(carla.Location(48, -64.7, 0.6), carla.Rotation(0, 180, 0)),
    # carla.Transform(carla.Location(-42, -30, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(-42, -10, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(-42, 5, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(-45.5, -25, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(-45.5, -5, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(-45.5, 10, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(106.5, -17, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(106.5, -5, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(106.5, 0, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(106.5, 10, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(110, -15, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(110, -5, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(110, 10, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(110, 5, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(106.5, 45, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(106.5, 55, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(106.5, 65, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(106.5, 75, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(110, 45, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(110, 55, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(110, 65, 0.6), carla.Rotation(0, -90, 0)),
    # carla.Transform(carla.Location(110, 75, 0.6), carla.Rotation(0, -90, 0))]
    # world.get_map().get_self.spawn_points()

    # actor vehicle 생성
    world.get_spectator().set_transform(spawn_points[0])
    vehicle = world.spawn_actor(bp, spawn_points[0])
    print(f'0: {spawn_points[0]}')
    # vehicle.set_autopilot(True)

    vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
    actor_list.append(vehicle)

    # 다른 vehicle 생성
    vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')

    print("vehicle spawning")
    for i in range(1, len(spawn_points)):
        tmp_vehicle = world.try_spawn_actor(vehicle_blueprints[i % len(vehicle_blueprints)], spawn_points[i])
        if tmp_vehicle is not None:
            # tmp_vehicle.set_autopilot(True)
            vehicle_list.append(tmp_vehicle)
        print(f'{i}: {spawn_points[i]}')

    # # walker 생성
    # print("walker spawning")
    # walker_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
    # walker_control_blueprint = world.get_blueprint_library().find('controller.ai.walker')
    # for i in range(50):
    #     spawn_point = carla.Transform()
    #     spawn_point.location = world.get_random_location_from_navigation()
    #     tmp_walker = world.try_spawn_actor(random.choice(walker_blueprints), spawn_point)
    #
    #     if tmp_walker is not None:
    #         tmp_controller = world.try_spawn_actor(walker_control_blueprint, carla.Transform(), attach_to=tmp_walker)
    #         walker_list.append(tmp_walker)
    #         walker_controller_list.append(tmp_controller)
    #
    # world.wait_for_tick()
    # for controller in walker_controller_list:
    #     controller.start()
    #     controller.go_to_location(world.get_random_location_from_navigation())
    #     controller.set_max_speed(1 + random.random())

    front_spawn_point = carla.Transform(carla.Location(x=2.5, z=0.4))
    left_spawn_point = carla.Transform(carla.Location(x=2.5, z=0.5), carla.Rotation(0, -45, 0))
    right_spawn_point = carla.Transform(carla.Location(x=2.5, z=0.6), carla.Rotation(0, 45, 0))
    # camera sensor 추가
    # cam_bp = bp_library.find("sensor.camera.rgb")
    # cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    # cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    # cam_bp.set_attribute("fov", "110")
    #
    #
    # rgb_sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    # actor_list.append(rgb_sensor)
    # rgb_sensor.listen(lambda data: process_img(data))

    # cam_dp_bp = bp_library.find("sensor.camera.depth")
    # cam_dp_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    # cam_dp_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    # cam_dp_bp.set_attribute("fov", "110")
    #
    # depth_sensor = world.spawn_actor(cam_dp_bp, spawn_point, attach_to=vehicle)
    # actor_list.append(depth_sensor)
    # depth_sensor.listen(lambda data: process_depth_img(data))

    # cam_sg_bp = bp_library.find("sensor.camera.semantic_segmentation")
    # cam_sg_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    # cam_sg_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    # cam_sg_bp.set_attribute("fov", "90")
    #
    # front_segmentation_sensor = world.spawn_actor(cam_sg_bp, front_spawn_point, attach_to=vehicle)
    # actor_list.append(front_segmentation_sensor)
    # front_segmentation_sensor.listen(lambda data: process_segmentation_img(data, 0))
    #
    # cam_sg_bp.set_attribute("fov", "90")
    # left_segmentation_sensor = world.spawn_actor(cam_sg_bp, left_spawn_point, attach_to=vehicle)
    # actor_list.append(left_segmentation_sensor)
    # left_segmentation_sensor.listen(lambda data: process_segmentation_img(data, 1))
    #
    # cam_sg_bp.set_attribute("fov", "90")
    # right_segmentation_sensor = world.spawn_actor(cam_sg_bp, right_spawn_point, attach_to=vehicle)
    # actor_list.append(right_segmentation_sensor)
    # right_segmentation_sensor.listen(lambda data: process_segmentation_img(data, 2))

    # line invasion sensor 추가
    line_sensor_bp = bp_library.find('sensor.other.lane_invasion')

    line_sensor = world.spawn_actor(line_sensor_bp, carla.Transform(), attach_to=vehicle)
    actor_list.append(line_sensor)
    line_sensor.listen(lambda event: on_invasion(event))

    radar_sensor_bp = bp_library.find('sensor.other.radar')
    radar_sensor_bp.set_attribute('horizontal_fov', '110')
    radar_sensor_bp.set_attribute('vertical_fov', '15')
    radar_sensor_bp.set_attribute('range', '8')
    rad_loc = carla.Location(x=2.0, z=1.0)
    rad_rot = carla.Rotation(pitch=5)
    radar_sensor = world.spawn_actor(radar_sensor_bp, carla.Transform(rad_loc, rad_rot), attach_to=vehicle)
    radar_sensor.listen(lambda data: radar_callback(data, vehicle))
    actor_list.append(radar_sensor)

    IMU_sensor_bp = bp_library.find('sensor.other.imu')
    IMU_sensor = world.spawn_actor(IMU_sensor_bp, carla.Transform(), attach_to=vehicle)
    IMU_sensor.listen(lambda data: IMU_callback(data))
    actor_list.append(IMU_sensor)


    cnt = 0
    throValue = 0.5
    handleValue = 0

    print('start')

    while 1:
        if cnt > 10000:
            break
        cnt += 1

        v = vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        if cnt > 100:
            handleValue = 0
        elif cnt > 75:
            handleValue = 0.1
        elif cnt > 50:
            handleValue = -0.1

        vehicle.apply_control(carla.VehicleControl(throttle=throValue, steer=handleValue))
        # print(f'kmh:{kmh} / throValue:{throValue}')

        waypoint = client.get_world().get_map().get_waypoint(vehicle.get_location(), project_to_road=True)

        vehicle_location = vehicle.get_location()
        waypoint_location = waypoint.transform.location
        distance = math.sqrt(math.pow((vehicle_location.x - waypoint_location.x), 2)
                             + math.pow((vehicle_location.y - waypoint_location.y), 2)
                             + math.pow((vehicle_location.z - waypoint_location.z), 2))

        vehicle_forward = vehicle.get_transform().get_forward_vector()
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

            # print(
            #     f'width {waypoint.lane_width} | left line type {waypoint.left_lane_marking} | left line distance {left_distance} '
            #     f'| right line type {waypoint.right_lane_marking.type} | right line distance {right_distance}')

        # print(f"distance: {distance} / deltaAngle: {deltaAngle}")

        #     action = random.randrange(0, 4)
        #     if action == 0:
        #         vehicle.apply_control(carla.VehicleControl(throttle=1, steer=1.0))
        #     elif action == 1:
        #         vehicle.apply_control(carla.VehicleControl(throttle=1, steer=-1.0))
        #     elif action == 2:
        #         vehicle.apply_control(carla.VehicleControl(throttle=1, reverse=True))
        #     else:
        #         vehicle.apply_control(carla.VehicleControl(throttle=1, steer=0.0))
        world.tick()


if __name__ == "__main__":

    try:
        main()

    finally:
        print(f'walker controller length: {len(walker_controller_list)}')
        for walcon in walker_controller_list:
            walcon.stop()
            walcon.destroy()

        print(f'walker length: {len(walker_list)}')
        client.apply_batch([carla.command.DestroyActor(x) for x in walker_list])

        print(f'actor length: {len(actor_list)}')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

        print(f'vehicle length: {len(vehicle_list)}')
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

        print("actors clean!")
