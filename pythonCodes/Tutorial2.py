import carla
import random
import time
import numpy as np
import cv2

import math

actor_list = []
vehicle_list = []
walker_list = []
walker_controller_list = []
IM_WIDTH = 640
IM_HEIGHT = 480

rgb_img = np.empty((IM_HEIGHT, IM_WIDTH, 3))
dep_img = np.empty((IM_HEIGHT, IM_WIDTH, 3))
seg_img = np.empty((IM_HEIGHT, IM_WIDTH, 3))


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


def process_segmentation_img(image):
    global seg_img
    image.convert(carla.ColorConverter.CityScapesPalette)
    i = np.array(image.raw_data)

    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    seg_img = i2[:, :, :3]

    render_image()
    return seg_img / 255.0


def render_image():
    # img = np.hstack((rgb_img, dep_img, seg_img))
    # cv2.imshow("sensors", img)
    cv2.imshow("sensors", rgb_img)
    cv2.waitKey(1)


def on_invasion(event):
    lane_types = set(x.type for x in event.crossed_lane_markings)
    # print(lane_types)
    # for x in lane_types:
    #     print(f'Crossed line {str(x)}')


def main(client):
    # environment 연결
    world = client.get_world()
    client.load_world('Town10HD')

    # world
    bp_library = world.get_blueprint_library()
    bp = bp_library.filter("model3")[0]

    spawn_points = []  # world.get_map().get_self.spawn_points()

    spawn_points.append(carla.Transform(carla.Location(-103, 0, 0.6), carla.Rotation(0, -90, 0)))  # actor spawn point

    spawn_points.append(carla.Transform(carla.Location(-103, -17, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(-106.5, -10, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(-90.2, -50, 0.6), carla.Rotation(0, -45, 0)))
    spawn_points.append(carla.Transform(carla.Location(-105.8, -30, 0.6), carla.Rotation(0, -80, 0)))

    spawn_points.append(carla.Transform(carla.Location(-30, -57.5, 0.6), carla.Rotation(0, 0, 0)))
    spawn_points.append(carla.Transform(carla.Location(30, -57.5, 0.6), carla.Rotation(0, 0, 0)))
    spawn_points.append(carla.Transform(carla.Location(50, -57.5, 0.6), carla.Rotation(0, 0, 0)))
    spawn_points.append(carla.Transform(carla.Location(-20, -60.5, 0.6), carla.Rotation(0, 0, 0)))
    spawn_points.append(carla.Transform(carla.Location(0, -60.5, 0.6), carla.Rotation(0, 0, 0)))
    spawn_points.append(carla.Transform(carla.Location(20, -60.5, 0.6), carla.Rotation(0, 0, 0)))

    spawn_points.append(carla.Transform(carla.Location(-110.1, -28, 0.6), carla.Rotation(0, 90, 0)))
    spawn_points.append(carla.Transform(carla.Location(-113.5, -23, 0.6), carla.Rotation(0, 90, 0)))
    spawn_points.append(carla.Transform(carla.Location(-106.5, -40, 0.6), carla.Rotation(0, 115, 0)))
    spawn_points.append(carla.Transform(carla.Location(-90.5, -63.5, 0.6), carla.Rotation(0, 150, 0)))

    spawn_points.append(carla.Transform(carla.Location(-20, -68.3, 0.6), carla.Rotation(0, 180, 0)))
    spawn_points.append(carla.Transform(carla.Location(0, -68.3, 0.6), carla.Rotation(0, 180, 0)))
    spawn_points.append(carla.Transform(carla.Location(40, -68.3, 0.6), carla.Rotation(0, 180, 0)))
    spawn_points.append(carla.Transform(carla.Location(-10, -64.7, 0.6), carla.Rotation(0, 180, 0)))
    spawn_points.append(carla.Transform(carla.Location(30, -64.7, 0.6), carla.Rotation(0, 180, 0)))
    spawn_points.append(carla.Transform(carla.Location(60, -64.7, 0.6), carla.Rotation(0, 180, 0)))

    spawn_points.append(carla.Transform(carla.Location(-42, -30, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(-42, -10, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(-42, 5, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(-45.5, -25, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(-45.5, -5, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(-45.5, 10, 0.6), carla.Rotation(0, -90, 0)))

    spawn_points.append(carla.Transform(carla.Location(106.5, -17, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(106.5, -5, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(106.5, 0, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(106.5, 10, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(110, -15, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(110, -5, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(110, 10, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(110, 5, 0.6), carla.Rotation(0, -90, 0)))

    spawn_points.append(carla.Transform(carla.Location(106.5, 45, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(106.5, 55, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(106.5, 65, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(106.5, 75, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(110, 45, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(110, 55, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(110, 65, 0.6), carla.Rotation(0, -90, 0)))
    spawn_points.append(carla.Transform(carla.Location(110, 75, 0.6), carla.Rotation(0, -90, 0)))

    # actor vehicle 생성
    world.get_spectator().set_transform(spawn_points[0])
    vehicle = world.spawn_actor(bp, spawn_points[0])
    print(f'0: {spawn_points[0]}')
    # vehicle.set_autopilot(True)

    vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
    actor_list.append(vehicle)

    # 다른 vehicle 생성
    vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
    #
    print("vehicle spawning")
    for i in range(1, len(spawn_points)):
        tmp_vehicle = world.try_spawn_actor(vehicle_blueprints[i%len(vehicle_blueprints)], spawn_points[i])
        if tmp_vehicle is not None:
            tmp_vehicle.set_autopilot(True)
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

    # camera sensor 추가
    cam_bp = bp_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")

    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    rgb_sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(rgb_sensor)
    rgb_sensor.listen(lambda data: process_img(data))

    cam_dp_bp = bp_library.find("sensor.camera.depth")
    cam_dp_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_dp_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_dp_bp.set_attribute("fov", "110")

    depth_sensor = world.spawn_actor(cam_dp_bp, spawn_point, attach_to=vehicle)
    actor_list.append(depth_sensor)
    depth_sensor.listen(lambda data: process_depth_img(data))

    cam_sg_bp = bp_library.find("sensor.camera.semantic_segmentation")
    cam_sg_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_sg_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_sg_bp.set_attribute("fov", "110")

    segmentation_sensor = world.spawn_actor(cam_sg_bp, spawn_point, attach_to=vehicle)
    actor_list.append(segmentation_sensor)
    segmentation_sensor.listen(lambda data: process_segmentation_img(data))

    # line invasion sensor 추가
    line_sensor_bp = bp_library.find('sensor.other.lane_invasion')

    line_sensor = world.spawn_actor(line_sensor_bp, carla.Transform(), attach_to=vehicle)
    actor_list.append(line_sensor)
    line_sensor.listen(lambda event: on_invasion(event))

    cnt = 0
    throValue = 0
    handleValue = 0

    while 1:
        if cnt > 10000:
            break
        cnt += 1

        v = vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        if cnt == 50:
            vehicle.apply_control(carla.VehicleControl(brake=1))

        if cnt > 400:
            if throValue <= 0.5:
                throValue += 0.1
            if handleValue >= -0.9:
                handleValue -= 0.1
        elif cnt > 200:
            if throValue >= 0.1:
                throValue -= 0.1
        elif cnt > 50:
            if throValue <= 0.5:
                throValue += 0.1

        # vehicle.apply_control(carla.VehicleControl(throttle=throValue, steer=handleValue))
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

    client = carla.Client('localhost', 2000)

    try:
        main(client)

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
