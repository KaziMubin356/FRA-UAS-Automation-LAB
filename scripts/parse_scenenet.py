
from PIL import Image
import math
import matplotlib
import numpy as np
import os
import pathlib
import random
import scenenet_pb2 as sn
import sys
import scipy.misc
import seaborn as sns
import glob
import matplotlib.pyplot as plt

NYU_13_CLASSES = [(0,'Unknown'),
                  (1,'Bed'),
                  (2,'Books'),
                  (3,'Ceiling'),
                  (4,'Chair'),
                  (5,'Floor'),
                  (6,'Furniture'),
                  (7,'Objects'),
                  (8,'Picture'),
                  (9,'Sofa'),
                  (10,'Table'),
                  (11,'TV'),
                  (12,'Wall'),
                  (13,'Window')
]

colour_code = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0.9137,0.3490,0.1882], #BOOKS
                       [0, 0.8549, 0], #CEILING
                       [0.5843,0,0.9412], #CHAIR
                       [0.8706,0.9451,0.0941], #FLOOR
                       [1.0000,0.8078,0.8078], #FURNITURE
                       [0,0.8784,0.8980], #OBJECTS
                       [0.4157,0.5333,0.8000], #PAINTING
                       [0.4588,0.1137,0.1608], #SOFA
                       [0.9412,0.1373,0.9216], #TABLE
                       [0,0.6549,0.6118], #TV
                       [0.9765,0.5451,0], #WALL
                       [0.8824,0.8980,0.7608]])

NYU_WNID_TO_CLASS = {
    '04593077':4, '03262932':4, '02933112':6, '03207941':7, '03063968':10, '04398044':7, '04515003':7,
    '00017222':7, '02964075':10, '03246933':10, '03904060':10, '03018349':6, '03786621':4, '04225987':7,
    '04284002':7, '03211117':11, '02920259':1, '03782190':11, '03761084':7, '03710193':7, '03367059':7,
    '02747177':7, '03063599':7, '04599124':7, '20000036':10, '03085219':7, '04255586':7, '03165096':1,
    '03938244':1, '14845743':7, '03609235':7, '03238586':10, '03797390':7, '04152829':11, '04553920':7,
    '04608329':10, '20000016':4, '02883344':7, '04590933':4, '04466871':7, '03168217':4, '03490884':7,
    '04569063':7, '03071021':7, '03221720':12, '03309808':7, '04380533':7, '02839910':7, '03179701':10,
    '02823510':7, '03376595':4, '03891251':4, '03438257':7, '02686379':7, '03488438':7, '04118021':5,
    '03513137':7, '04315948':7, '03092883':10, '15101854':6, '03982430':10, '02920083':1, '02990373':3,
    '03346455':12, '03452594':7, '03612814':7, '06415419':7, '03025755':7, '02777927':12, '04546855':12,
    '20000040':10, '20000041':10, '04533802':7, '04459362':7, '04177755':9, '03206908':7, '20000021':4,
    '03624134':7, '04186051':7, '04152593':11, '03643737':7, '02676566':7, '02789487':6, '03237340':6,
    '04502670':7, '04208936':7, '20000024':4, '04401088':7, '04372370':12, '20000025':4, '03956922':7,
    '04379243':10, '04447028':7, '03147509':7, '03640988':7, '03916031':7, '03906997':7, '04190052':6,
    '02828884':4, '03962852':1, '03665366':7, '02881193':7, '03920867':4, '03773035':12, '03046257':12,
    '04516116':7, '00266645':7, '03665924':7, '03261776':7, '03991062':7, '03908831':7, '03759954':7,
    '04164868':7, '04004475':7, '03642806':7, '04589593':13, '04522168':7, '04446276':7, '08647616':4,
    '02808440':7, '08266235':10, '03467517':7, '04256520':9, '04337974':7, '03990474':7, '03116530':6,
    '03649674':4, '04349401':7, '01091234':7, '15075141':7, '20000028':9, '02960903':7, '04254009':7,
    '20000018':4, '20000020':4, '03676759':11, '20000022':4, '20000023':4, '02946921':7, '03957315':7,
    '20000026':4, '20000027':4, '04381587':10, '04101232':7, '03691459':7, '03273913':7, '02843684':7,
    '04183516':7, '04587648':13, '02815950':3, '03653583':6, '03525454':7, '03405725':6, '03636248':7,
    '03211616':11, '04177820':4, '04099969':4, '03928116':7, '04586225':7, '02738535':4, '20000039':10,
    '20000038':10, '04476259':7, '04009801':11, '03909406':12, '03002711':7, '03085602':11, '03233905':6,
    '20000037':10, '02801938':7, '03899768':7, '04343346':7, '03603722':7, '03593526':7, '02954340':7,
    '02694662':7, '04209613':7, '02951358':7, '03115762':9, '04038727':6, '03005285':7, '04559451':7,
    '03775636':7, '03620967':10, '02773838':7, '20000008':6, '04526964':7, '06508816':7, '20000009':6,
    '03379051':7, '04062428':7, '04074963':7, '04047401':7, '03881893':13, '03959485':7, '03391301':7,
    '03151077':12, '04590263':13, '20000006':1, '03148324':6, '20000004':1, '04453156':7, '02840245':2,
    '04591713':7, '03050864':7, '03727837':5, '06277280':11, '03365592':5, '03876519':8, '03179910':7,
    '06709442':7, '03482252':7, '04223580':7, '02880940':7, '04554684':7, '20000030':9, '03085013':7,
    '03169390':7, '04192858':7, '20000029':9, '04331277':4, '03452741':7, '03485997':7, '20000007':1,
    '02942699':7, '03231368':10, '03337140':7, '03001627':4, '20000011':6, '20000010':6, '20000013':6,
    '04603729':10, '20000015':4, '04548280':12, '06410904':2, '04398951':10, '03693474':9, '04330267':7,
    '03015149':9, '04460038':7, '03128519':7, '04306847':7, '03677231':7, '02871439':6, '04550184':6,
    '14974264':7, '04344873':9, '03636649':7, '20000012':6, '02876657':7, '03325088':7, '04253437':7,
    '02992529':7, '03222722':12, '04373704':4, '02851099':13, '04061681':10, '04529681':7,
}


def photo_path_from_view(render_path,view):
    photo_path = os.path.join(render_path,'photo')
    image_path = os.path.join(photo_path,'{0}.jpg'.format(view.frame_num))
    return os.path.join(data_root_path,image_path)

def instance_path_from_view(render_path,view):
    photo_path = os.path.join(render_path,'instance')
    image_path = os.path.join(photo_path,'{0}.png'.format(view.frame_num))
    return os.path.join(data_root_path,image_path)

def save_class_from_instance(instance_path,class_path, class_NYUv2_colourcode_path, mapping):
    instance_img = np.asarray(Image.open(instance_path))
    class_img = np.zeros(instance_img.shape)
    h,w  = instance_img.shape

    for instance, semantic_class in mapping.items():
        class_img[instance_img == instance] = semantic_class

    return class_img
def normalize(v):
    return v/np.linalg.norm(v)

def load_depth_map_in_m(file_name):
    image = Image.open(file_name)
    pixel = np.array(image)
    return (pixel * 0.001)

def pixel_to_ray(pixel,vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    x, y = pixel
    x_vect = math.tan(math.radians(hfov/2.0)) * ((2.0 * ((x+0.5)/pixel_width)) - 1.0)
    y_vect = math.tan(math.radians(vfov/2.0)) * ((2.0 * ((y+0.5)/pixel_height)) - 1.0)
    return (x_vect,y_vect,1.0)

def normalised_pixel_to_ray_array(width=320,height=240):
    pixel_to_ray_array = np.zeros((height,width,3))
    for y in range(height):
        for x in range(width):
            pixel_to_ray_array[y,x] = normalize(np.array(pixel_to_ray((x,y),pixel_height=height,pixel_width=width)))
    return pixel_to_ray_array

def points_in_camera_coords(depth_map,pixel_to_ray_array):
    assert depth_map.shape[0] == pixel_to_ray_array.shape[0]
    assert depth_map.shape[1] == pixel_to_ray_array.shape[1]
    assert len(depth_map.shape) == 2
    assert pixel_to_ray_array.shape[2] == 3
    camera_relative_xyz = np.ones((depth_map.shape[0],depth_map.shape[1],4))
    for i in range(3):
        camera_relative_xyz[:,:,i] = depth_map * pixel_to_ray_array[:,:,i]
    return camera_relative_xyz

def flatten_points(points):
    return points.reshape(-1, 4)

def reshape_points(height,width,points):
    other_dim = points.shape[1]
    return points.reshape(height,width,other_dim)

def transform_points(transform,points):
    assert points.shape[2] == 4
    height = points.shape[0]
    width = points.shape[1]
    points = flatten_points(points)
    return reshape_points(height,width,(transform.dot(points.T)).T)

def world_to_camera_with_pose(view_pose):
    lookat_pose = position_to_np_array(view_pose.lookat)
    camera_pose = position_to_np_array(view_pose.camera)
    up = np.array([0,1,0])
    R = np.diag(np.ones(4))
    R[2,:3] = normalize(lookat_pose - camera_pose)
    R[0,:3] = normalize(np.cross(R[2,:3],up))
    R[1,:3] = -normalize(np.cross(R[0,:3],R[2,:3]))
    T = np.diag(np.ones(4))
    T[:3,3] = -camera_pose
    return R.dot(T)

def camera_to_world_with_pose(view_pose):
    return np.linalg.inv(world_to_camera_with_pose(view_pose))

def camera_point_to_uv_pixel_location(point,vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    point = point / point[2]
    u = ((pixel_width/2.0) * ((point[0]/math.tan(math.radians(hfov/2.0))) + 1))
    v = ((pixel_height/2.0) * ((point[1]/math.tan(math.radians(vfov/2.0))) + 1))
    return (u,v)

def position_to_np_array(position):
    return np.array([position.x,position.y,position.z])

def interpolate_poses(start_pose,end_pose,alpha):
    assert alpha >= 0.0
    assert alpha <= 1.0
    camera_pose = alpha * position_to_np_array(end_pose.camera)
    camera_pose += (1.0 - alpha) * position_to_np_array(start_pose.camera)
    lookat_pose = alpha * position_to_np_array(end_pose.lookat)
    lookat_pose += (1.0 - alpha) * position_to_np_array(start_pose.lookat)
    timestamp = alpha * end_pose.timestamp + (1.0 - alpha) * start_pose.timestamp
    pose = sn.Pose()
    pose.camera.x = camera_pose[0]
    pose.camera.y = camera_pose[1]
    pose.camera.z = camera_pose[2]
    pose.lookat.x = lookat_pose[0]
    pose.lookat.y = lookat_pose[1]
    pose.lookat.z = lookat_pose[2]
    pose.timestamp = timestamp
    return pose

def world_point_to_uv_pixel_location_with_interpolated_camera(point,shutter_open,shutter_close,alpha):
    view_pose = interpolate_poses(shutter_open,shutter_close,alpha)
    wTc = world_to_camera_with_pose(view_pose)
    point_in_camera_coords = wTc.dot(np.array(point))
    uv = camera_point_to_uv_pixel_location(point_in_camera_coords)
    return uv

# Expects:
# an nx4 array of points of the form [[x,y,z,1],[x,y,z,1]...] in world coordinates
# a length three array [x,y,z] for camera start and end (of a shutter) and lookat start/end in world coordinates

# Returns:
# a nx2 array of the horizontal and vertical pixel location time derivatives (i.e. pixels per second in the horizontal and vertical)
# NOTE: the pixel coordinates are defined as (0,0) in the top left corner, to (320,240) in the bottom left
def optical_flow(points,shutter_open,shutter_close,alpha=0.5,shutter_time=(1.0/60),
                 hfov=60,pixel_width=320,vfov=45,pixel_height=240):
    # Alpha is the linear interpolation coefficient, 0.5 takes the derivative in the midpoint
    # which is where the ground truth renders are taken.  The photo render integrates via sampling
    # over the whole shutter open-close trajectory
    view_pose = interpolate_poses(shutter_open,shutter_close,alpha)
    wTc = world_to_camera_with_pose(view_pose)
    camera_pose = position_to_np_array(view_pose.camera)
    lookat_pose = position_to_np_array(view_pose.lookat)

    # Get camera pixel scale constants
    uk = (pixel_width/2.0) * ((1.0/math.tan(math.radians(hfov/2.0))))
    vk = (pixel_height/2.0) * ((1.0/math.tan(math.radians(vfov/2.0))))

    # Get basis vectors
    ub1 = lookat_pose - camera_pose
    b1 = normalize(ub1)
    ub2 = np.cross(b1,np.array([0,1,0]))
    b2 = normalize(ub2)
    ub3 = np.cross(b2,b1)
    b3 = -normalize(ub3)

    # Get camera pose alpha derivative
    camera_end = position_to_np_array(shutter_close.camera)
    camera_start = position_to_np_array(shutter_open.camera)
    lookat_end = position_to_np_array(shutter_close.lookat)
    lookat_start= position_to_np_array(shutter_open.lookat)
    dc_dalpha = camera_end - camera_start

    # Get basis vector derivatives
    # dub1 means d unnormalised b1
    db1_dub1 = (np.eye(3) - np.outer(b1,b1))/np.linalg.norm(ub1)
    dub1_dalpha = lookat_end - lookat_start - camera_end + camera_start
    db1_dalpha = db1_dub1.dot(dub1_dalpha)
    db2_dub2 = (np.eye(3) - np.outer(b2,b2))/np.linalg.norm(ub2)
    dub2_dalpha = np.array([-db1_dalpha[2],0,db1_dalpha[0]])
    db2_dalpha = db2_dub2.dot(dub2_dalpha)
    db3_dub3 = (np.eye(3) - np.outer(b3,b3))/np.linalg.norm(ub3)
    dub3_dalpha = np.array([
            -(db2_dalpha[2]*b1[1]+db1_dalpha[1]*b2[2]),
            -(db2_dalpha[0]*b1[2] + db1_dalpha[2]*b2[0])+(db2_dalpha[2]*b1[0]+db1_dalpha[0]*b2[2]),
            (db1_dalpha[1]*b2[0]+db2_dalpha[0]*b1[1])
        ])
    db3_dalpha = -db3_dub3.dot(dub3_dalpha)

    # derivative of the rotated translation offset
    dt3_dalpha = np.array([
            -db2_dalpha.dot(camera_pose)-dc_dalpha.dot(b2),
            -db3_dalpha.dot(camera_pose)-dc_dalpha.dot(b3),
            -db1_dalpha.dot(camera_pose)-dc_dalpha.dot(b1),
        ])

    # camera transform derivative
    dT_dalpha = np.empty((4,4))
    dT_dalpha[0,:3] = db2_dalpha
    dT_dalpha[1,:3] = db3_dalpha
    dT_dalpha[2,:3] = db1_dalpha
    dT_dalpha[:3,3] = dt3_dalpha

    # Calculate 3D point derivative alpha derivative
    dpoint_dalpha = dT_dalpha.dot(points.T)
    point_in_camera_coords = wTc.dot(np.array(points.T))

    # Calculate pixel location alpha derivative
    du_dalpha = uk * (dpoint_dalpha[0] * point_in_camera_coords[2] - dpoint_dalpha[2] * point_in_camera_coords[0])
    dv_dalpha = vk * (dpoint_dalpha[1] * point_in_camera_coords[2] - dpoint_dalpha[2] * point_in_camera_coords[1])
    du_dalpha = du_dalpha/(point_in_camera_coords[2]*point_in_camera_coords[2])
    dv_dalpha = dv_dalpha/(point_in_camera_coords[2]*point_in_camera_coords[2])

    # Calculate pixel location time derivative
    du_dt = du_dalpha / shutter_time
    dv_dt = dv_dalpha / shutter_time
    return np.vstack((du_dt,dv_dt)).T

def flow_to_hsv_image(flow, magnitude_scale=1.0/100.0):
    hsv = np.empty((240,320,3))
    for row in range(240):
        for col in range(320):
            v = flow[row,col,:]
            magnitude = np.linalg.norm(v)
            if magnitude < 1e-8:
                hsv[row,col,0] = 0.0
                hsv[row,col,1] = 0.0
                hsv[row,col,2] = 0.0
            else:
                direction = v / magnitude
                theta = math.atan2(direction[1], direction[0])
                if theta <= 0:
                    theta += 2*math.pi
                assert(theta >= 0.0 and theta <= 2*math.pi)
                hsv[row,col,0] = theta / (2*math.pi)
                hsv[row,col,1] = 1.0
                hsv[row,col,2] = min(magnitude * magnitude_scale, 1.0)
    return hsv


def depth_path_from_view(render_path,view):
    photo_path = os.path.join(render_path,'depth')
    depth_path = os.path.join(photo_path,'{0}.png'.format(view.frame_num))
    return os.path.join(data_root_path,depth_path)

os.system('make')
os.system('wget https://www.doc.ic.ac.uk/~bjm113/scenenet_data/train_protobufs.tar.gz')
os.sysem('tar -xf train_protobufs.tar.gz')
os.system('mkdir train')

##Scenenet dataset is separated into 17 tarballs
for i in range(0,17):
    os.system(f'wget https://www.doc.ic.ac.uk/~bjm113/scenenet_data/train_split/train_{i}.tar.gz')
    os.system(f'tar -xf train_{i}.tar.gz')
    os.system(f'mkdir ./{i}')
    os.system(f'mkdir ./{i}/x')
    os.system(f'mkdir ./{i}/y')
    os.system(f'mkdir ./{i}/z')
    os.system(f'mkdir ./{i}/rgb')
    os.system(f'mkdir ./{i}/label')
    data_root_path = './train'
    protobuf_path = f'./train_protobufs/scenenet_rgbd_train_{i}.pb'
    framenum=0
    trajectories = sn.Trajectories()
    try:
        with open(protobuf_path,'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Please ensure you have copied the pb file to the data directory')

    # This stores for each image pixel, the cameras 3D ray vector 
    cached_pixel_to_ray_array = normalised_pixel_to_ray_array()
    for traj in trajectories.trajectories:
        instance_class_map = {}
        for instance in traj.instances:
            instance_type = sn.Instance.InstanceType.Name(instance.instance_type)
            if instance.instance_type != sn.Instance.BACKGROUND:
                instance_class_map[instance.instance_id] = NYU_WNID_TO_CLASS[instance.semantic_wordnet_id]
        for idx,view in enumerate(traj.views):
            depth_path = depth_path_from_view(traj.render_path,view)
            name = os.path.split(depth_path)[-1][:-4]
            if int(name)%250==0: 
                folder = depth_path.split('/')[-3]
                name = name+folder
                optical_flow_path = 'optical_flow_{0}.png'.format(idx)
                depth_map = load_depth_map_in_m(str(depth_path))
                depth_map[depth_map == 0.0] = 1000.0
                points_in_camera = points_in_camera_coords(depth_map,cached_pixel_to_ray_array)
                instance_path = instance_path_from_view(traj.render_path,view)
                rgb_path = photo_path_from_view(traj.render_path,view)
                rgb = Image.open(rgb_path)
                label = save_class_from_instance(instance_path,'semantic_class.png','NYUv2.png',instance_class_map)

                # Transform point from camera coordinates into world coordinates
                ground_truth_pose = interpolate_poses(view.shutter_open,view.shutter_close,0.5)
                camera_to_world_matrix = camera_to_world_with_pose(ground_truth_pose)
                points_in_world = transform_points(camera_to_world_matrix,points_in_camera)
                points_in_world = points_in_camera
                points_in_world = np.array(points_in_world)

                x = points_in_world[:,:,0]
                y = points_in_world[:,:,1]
                z = points_in_world[:,:,2]
                
                
                x = (x-x.min())/(x.max()-x.min()+1e-7)*255
                y = (y-y.min())/(y.max()-y.min()+1e-7)*255
                z = (z-z.min())/(z.max()-z.min()+1e-7)*255
                Image.fromarray(x.astype(np.uint8)).save(f'./{i}/x/{name}.jpg')
                Image.fromarray(y.astype(np.uint8)).save(f'./{i}/y/{name}.jpg')
                Image.fromarray(z.astype(np.uint8)).save(f'./{i}/z/{name}.jpg')
                rgb.save(f'./{i}/rgb/{name}.jpg')
                Image.fromarray(label.astype(np.uint8)).save(f'./{i}/label/{name}.png')

                if framenum%5000==0:
                    print(framenum)
                framenum+=1 


os.system(f'mkdir ./data/')
os.system(f'mkdir ./data/x')
os.system(f'mkdir ./data/y')
os.system(f'mkdir ./data/z')
os.system(f'mkdir ./data/rgb')
os.system(f'mkdir ./data/label')
data_root_path = './val'
protobuf_path = f'./scenenet_rgbd_val.pb'
framenum=0
trajectories = sn.Trajectories()
try:
        with open(protobuf_path,'rb') as f:
            trajectories.ParseFromString(f.read())
except IOError:
        print('Scenenet protobuf data not found at location:{0}'.format(data_root_path))
        print('Please ensure you have copied the pb file to the data directory')

    # This stores for each image pixel, the cameras 3D ray vector 
cached_pixel_to_ray_array = normalised_pixel_to_ray_array()
for traj in trajectories.trajectories:
        instance_class_map = {}
        for instance in traj.instances:
            instance_type = sn.Instance.InstanceType.Name(instance.instance_type)
            if instance.instance_type != sn.Instance.BACKGROUND:
                instance_class_map[instance.instance_id] = NYU_WNID_TO_CLASS[instance.semantic_wordnet_id]
        for idx,view in enumerate(traj.views):
            depth_path = depth_path_from_view(traj.render_path,view)
            #print(depth_path)
            name = os.path.split(depth_path)[-1][:-4]
            if int(name)%250==0: 
                folder = depth_path.split('/')[-3]
                name = name+folder
                optical_flow_path = 'optical_flow_{0}.png'.format(idx)
                depth_map = load_depth_map_in_m(str(depth_path))
                depth_map[depth_map == 0.0] = 1000.0
                points_in_camera = points_in_camera_coords(depth_map,cached_pixel_to_ray_array)
                instance_path = instance_path_from_view(traj.render_path,view)
                rgb_path = photo_path_from_view(traj.render_path,view)
                rgb = Image.open(rgb_path)
                label = save_class_from_instance(instance_path,'semantic_class.png','NYUv2.png',instance_class_map)

                # Transform point from camera coordinates into world coordinates
                ground_truth_pose = interpolate_poses(view.shutter_open,view.shutter_close,0.5)
                camera_to_world_matrix = camera_to_world_with_pose(ground_truth_pose)
                points_in_world = transform_points(camera_to_world_matrix,points_in_camera)
                points_in_world = points_in_camera
                points_in_world = np.array(points_in_world)

                x = points_in_world[:,:,0]
                y = points_in_world[:,:,1]
                z = points_in_world[:,:,2]
                
                x[x>5] = -5
                y[y>5] = -5
                z[z>20] = 0
                
                x[x<-5] = -5
                y[y<-5] = -5
                
                x = (x-x.min())/(x.max()-x.min()+1e-7)*255
                y = (y-y.min())/(y.max()-y.min()+1e-7)*255
                z = (z-z.min())/(z.max()-z.min()+1e-7)*255

                Image.fromarray(x.astype(np.uint8)).save(f'./data/x/{name}.jpg')
                Image.fromarray(y.astype(np.uint8)).save(f'./data/y/{name}.jpg')
                Image.fromarray(z.astype(np.uint8)).save(f'./data/z/{name}.jpg')
                rgb.save(f'./data/rgb/{name}.jpg')
                Image.fromarray(np.array(label).astype(np.uint8)).save(f'./data/label/{name}.png')

                if framenum%5000==0:
                    print(framenum)
                framenum+=1 
                
                
