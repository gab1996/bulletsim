#include <pcl/ros/conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <cv_bridge/cv_bridge.h>
#include <cv.h>
#include <bulletsim_msgs/TrackedObject.h>
#include "clouds/utils_ros.h"

#include "clouds/utils_pcl.h"
#include "utils_tracking.h"
#include "utils/logging.h"
#include "utils/utils_vector.h"
#include "visibility.h"
#include "physics_tracker.h"
#include "feature_extractor.h"
#include "initialization.h"
#include "simulation/simplescene.h"
#include "config_tracking.h"
#include "utils/conversions.h"
#include "clouds/cloud_ops.h"
#include "simulation/util.h"
#include "clouds/utils_cv.h"
#include "simulation/recording.h"
#include "cam_sync.h"

#include "simulation/config_viewer.h"
#include <geometry_msgs/PoseArray.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
// #include <tf2_msgs/TFMessage.h>

using sensor_msgs::PointCloud2;
using sensor_msgs::Image;
using namespace std;
using namespace Eigen;

namespace cv {
	typedef Vec<uchar, 3> Vec3b;
}

int nCameras;

vector<cv::Mat> rgb_images;
vector<cv::Mat> mask_images;
vector<cv::Mat> depth_images;
vector<CoordinateTransformer*> transformers;

ColorCloudPtr filteredCloud(new ColorCloud()); // filtered cloud in ground frame
bool pending = false; // new message received, waiting to be processed
tf::TransformListener* listener;

int dio_counter=0;
int gesu_counter=1;

void callback(const vector<sensor_msgs::PointCloud2ConstPtr>& cloud_msg, const vector<sensor_msgs::ImageConstPtr>& image_msgs) {

  dio_counter ++;
  if (rgb_images.size()!=nCameras) rgb_images.resize(nCameras);
  if (mask_images.size()!=nCameras) mask_images.resize(nCameras);
  if (depth_images.size()!=nCameras) depth_images.resize(nCameras);
  assert(image_msgs.size() == 2*nCameras);
  for (int i=0; i<nCameras; i++) {
  	// merge all the clouds progressively
  	ColorCloudPtr cloud(new ColorCloud);
    //std::cout<<"andamo1"<<std::endl;
		pcl::fromROSMsg(*cloud_msg[i], *cloud);
    //std::cout<<"andamo"<<std::endl;
		pcl::transformPointCloud(*cloud, *cloud, transformers[i]->worldFromCamEigen);
    
  

		if (i==0) *filteredCloud = *cloud;
  	else *filteredCloud = *filteredCloud + *cloud;

		extractImageAndMask(cv_bridge::toCvCopy(image_msgs[2*i],"8UC4")->image, rgb_images[i], mask_images[i]);
  	depth_images[i] = cv_bridge::toCvCopy(image_msgs[2*i+1],"32FC1")->image;
  }

  filteredCloud = downsampleCloud(filteredCloud, TrackingConfig::downsample*METERS);


  //filteredCloud = filterZ(filteredCloud, -0.1*METERS, 0.20*METERS);

  pending = true;
}

int main(int argc, char* argv[]) {
  Eigen::internal::setNbThreads(2);


  rosbag::Bag bag;
  bag.open("/usr/ws/sollevamento100_1_filtered.bag", rosbag::bagmode::Read);
  std::vector<sensor_msgs::PointCloud2ConstPtr> pc_vec;
  std::vector<sensor_msgs::ImageConstPtr> imagevec;
  std::vector<sensor_msgs::ImageConstPtr> depthvec;
  // std::vector<tf2_msgs::TFMessageConstPtr> tfvec;
  rosbag::View view(bag);


 for (rosbag::View::iterator it = view.begin(); it != view.end(); ++it)
        {
        rosbag::MessageInstance const& msg = *it;
        if (msg.getTopic() == "/preprocessor/kinect1/points")
        {
            sensor_msgs::PointCloud2ConstPtr pcMsg = msg.instantiate<sensor_msgs::PointCloud2>();
            pc_vec.push_back(pcMsg);

        }
        else if (msg.getTopic() == "/preprocessor/kinect1/image")
        {
            sensor_msgs::ImageConstPtr imageMsg = msg.instantiate<sensor_msgs::Image>();
            imagevec.push_back(imageMsg);
        }
        else if (msg.getTopic() == "/preprocessor/kinect1/depth")
        {
            sensor_msgs::ImageConstPtr depthMsg = msg.instantiate<sensor_msgs::Image>();
            depthvec.push_back(depthMsg);
        }
        // else if (msg.getTopic() == "/tf_static")
        // {
        //     tf2_msgs::TFMessageConstPtr tfMsg = msg.instantiate<tf2_msgs::TFMessage>();
        //     tfvec.push_back(tfMsg);
        // }
    }
  bag.close();
  // std::cout<<"daje"<<std::endl;
  // std::cout<<pc_vec.size()<<std::endl;
  // std::cout<<imagevec.size()<<std::endl;
  // std::cout<<depthvec.size()<<std::endl;


  GeneralConfig::scale = 1;
  BulletConfig::maxSubSteps = 0;
  // BulletConfig::gravity = btVector3(0,0,-0.01);
  BulletConfig::gravity = btVector3(0,0,0);
  Parser parser;
  parser.addGroup(TrackingConfig());
  parser.addGroup(GeneralConfig());
  parser.addGroup(BulletConfig());
  parser.addGroup(ViewerConfig());
  parser.addGroup(RecordingConfig());
  parser.read(argc, argv);

  nCameras = TrackingConfig::cameraTopics.size();

  ros::init(argc, argv,"tracker_node");
  ros::NodeHandle nh;

  // listener = new tf::TransformListener();
  //tf::TransformListener listener;
  // for (int i=0; i<nCameras; i++)
   	//transformers.push_back(new CoordinateTransformer(waitForAndGetTransform(*listener, "/ground", TrackingConfig::cameraTopics[i]+"_rgb_optical_frame")));

  tf::StampedTransform s_mio;
  // // btTransform()
  // s_mio.frame_id_="/ground";
  // s_mio.child_frame_id_="/kinect1_rgb_optical_frame";
  // s_mio.stamp_=ros::Time(0);
  // s_mio.setOrigin(tf::Vector3(0.643, 0.025, 0.650));
  // s_mio.setRotation(tf::Quaternion(0.837,0.224,-0.129,-0.483));

  s_mio.frame_id_="/ground";
  s_mio.child_frame_id_="/kinect1_rgb_optical_frame";
  s_mio.stamp_=ros::Time(0);
  s_mio.setOrigin(tf::Vector3(-0.209, 0.129, 1.561));
  s_mio.setRotation(tf::Quaternion(-0.645, 0.698, -0.230, 0.210));

  //listener.waitForTransform(s_mio.frame_id_,"/kinect1_rgb_optical_frame", ros::Time(0),ros::Duration(.1));
	//listener.lookupTransform(s_mio.frame_id_,"/kinect1_rgb_optical_frame", ros::Time(0), s_mio);
  transformers.push_back(new CoordinateTransformer(s_mio.asBt()));
  //cout<<transformers[0]->camFromWorldEigen.rotation()<<endl;
	vector<string> cloud_topics;
	vector<string> image_topics;
  for (int i=0; i<nCameras; i++) {
		cloud_topics.push_back("/preprocessor" + TrackingConfig::cameraTopics[i] + "/points");
		image_topics.push_back("/preprocessor" + TrackingConfig::cameraTopics[i] + "/image");
		image_topics.push_back("/preprocessor" + TrackingConfig::cameraTopics[i] + "/depth");
  }

	//synchronizeAndRegisterCallback(cloud_topics, image_topics, nh, callback);
  //cout<<METERS<<endl;

  ros::Publisher objPub = nh.advertise<bulletsim_msgs::TrackedObject>(trackedObjectTopic,1000,true);
  ros::Publisher pc_Pub = nh.advertise<sensor_msgs::PointCloud2>("/preprocessor/kinect1/points",1000,true);
  ros::Publisher vel_pub = nh.advertise<geometry_msgs::PoseArray>("/vel_topic", 1000);
  

  //std::cout<<"dio"<<std::endl;
  // ros::topic::waitForMessage<PointCloud2>("/preprocessor/kinect1/points",nh);
  //std::cout<<"cane"<<std::endl;
  // wait for first message, then initialize
  // while (!pending) {
  //   ros::spinOnce();
  //   sleep(.001);
  //   if (!ros::ok()) throw runtime_error("caught signal while waiting for first message");
  // }

  // set up scene
  Scene scene;
  util::setGlobalEnv(scene.env);

  if (TrackingConfig::record_camera_pos_file != "" &&
      TrackingConfig::playback_camera_pos_file != "") {
    throw runtime_error("can't both record and play back camera positions");
  }
  CamSync camsync(scene);
  if (TrackingConfig::record_camera_pos_file != "") {
    camsync.enable(CamSync::RECORD, TrackingConfig::record_camera_pos_file);
  } else if (TrackingConfig::playback_camera_pos_file != "") {
    camsync.enable(CamSync::PLAYBACK, TrackingConfig::playback_camera_pos_file);
  }
  // cout<<transformers[0]->camFromWorldUnscaled.getOrigin().getX()<<endl;
  // cout<<transformers[0]->camFromWorldUnscaled.getOrigin().getY()<<endl;
  // cout<<transformers[0]->camFromWorldUnscaled.getOrigin().getZ()<<endl;
  // cout<<transformers[0]->worldFromCamUnscaled.getRotation()<<endl;
  // cout<<transformers[0]->worldFromCamUnscaled.getOrigin()<<endl;
  // cout<<transformers[0]->worldFromCamUnscaled.getOpenGLMatrix()<<endl;
  // ROS_INFO("kl: %f",TrackingConfig::kl_insole);
	// ROS_INFO("n: %d",TrackingConfig::n_nodes_bending);
	// ROS_INFO("mass: %f",TrackingConfig::mass_insole);
  // ROS_INFO("kp: %f",TrackingConfig::kp_insole);
	// ROS_INFO("kd: %f",TrackingConfig::kd_insole);
  
  ViewerConfig::cameraHomePosition = transformers[0]->worldFromCamUnscaled.getOrigin();
  ViewerConfig::cameraHomeCenter = ViewerConfig::cameraHomePosition + transformers[0]->worldFromCamUnscaled.getBasis().getColumn(2);
  ViewerConfig::cameraHomeUp = -transformers[0]->worldFromCamUnscaled.getBasis().getColumn(1);
  scene.startViewer();

  vector<sensor_msgs::PointCloud2ConstPtr> cloud_mia(1);
  vector<sensor_msgs::ImageConstPtr> image_mia(2);
  cloud_mia[0]=pc_vec[0];
  image_mia[0]=imagevec[0];
  image_mia[1]=depthvec[0];
  // pc_Pub.publish(cloud_mia[0]);
  callback(cloud_mia,image_mia);
	TrackedObject::Ptr trackedObj = callInitServiceAndCreateObject(filteredCloud, rgb_images[0], mask_images[0], transformers[0]);
  if (!trackedObj) throw runtime_error("initialization of object failed.");
  

  trackedObj->init();
  //std::cout<<"provaa"<<endl;
  scene.env->add(trackedObj->m_sim);

 	// actual tracking algorithm
	MultiVisibility::Ptr visInterface(new MultiVisibility());
	for (int i=0; i<nCameras; i++) {
		if (trackedObj->m_type == "rope") // Don't do self-occlusion if the trackedObj is a rope
			visInterface->addVisibility(DepthImageVisibility::Ptr(new DepthImageVisibility(transformers[i])));
		else
			visInterface->addVisibility(AllOcclusionsVisibility::Ptr(new AllOcclusionsVisibility(scene.env->bullet->dynamicsWorld, transformers[i])));
	}

	TrackedObjectFeatureExtractor::Ptr objectFeatures(new TrackedObjectFeatureExtractor(trackedObj));
	CloudFeatureExtractor::Ptr cloudFeatures(new CloudFeatureExtractor());
	PhysicsTracker::Ptr alg(new PhysicsTracker(objectFeatures, cloudFeatures, visInterface));
	PhysicsTrackerVisualizer::Ptr trackingVisualizer(new PhysicsTrackerVisualizer(&scene, alg));

	bool applyEvidence = true;
  scene.addVoidKeyCallback('a',boost::bind(toggle, &applyEvidence), "apply evidence");
  scene.addVoidKeyCallback('=',boost::bind(&EnvironmentObject::adjustTransparency, trackedObj->getSim(), 0.1f), "increase opacity");
  scene.addVoidKeyCallback('-',boost::bind(&EnvironmentObject::adjustTransparency, trackedObj->getSim(), -0.1f), "decrease opacity");
  bool exit_loop = false;
  scene.addVoidKeyCallback('q',boost::bind(toggle, &exit_loop), "exit");

  boost::shared_ptr<ScreenThreadRecorder> screen_recorder;
  boost::shared_ptr<ImageTopicRecorder> image_topic_recorder;
  if (RecordingConfig::record == RECORD_RENDER_ONLY) {
		screen_recorder.reset(new ScreenThreadRecorder(scene.viewer, RecordingConfig::dir + "/" +  RecordingConfig::video_file + "_tracked.avi"));
  } else if (RecordingConfig::record == RECORD_RENDER_AND_TOPIC) {
		screen_recorder.reset(new ScreenThreadRecorder(scene.viewer, RecordingConfig::dir + "/" +  RecordingConfig::video_file + "_tracked.avi"));
		image_topic_recorder.reset(new ImageTopicRecorder(nh, image_topics[0], RecordingConfig::dir + "/" +  RecordingConfig::video_file + "_topic.avi"));
  }

  scene.setSyncTime(false);
  scene.setDrawing(true);
  ros::Rate rate(30);
  // while (!exit_loop && ros::ok()) {
  
  // objPub.publish(toTrackedObjectMessage(trackedObj));  
  // pc_Pub.publish(pc_vec[0]);



    sleep(1);
    
    

  
  for (int i=1;i<pc_vec.size();i++){
  //Update the inputs of the featureExtractors and visibilities (if they have any inputs)
  cloudFeatures->updateInputs(filteredCloud, rgb_images[0], transformers[0]);
  for (int j=0; j<nCameras; j++)
  	visInterface->visibilities[j]->updateInput(depth_images[j]);
    // pending = false;
  
    
  // while (ros::ok() && !pending) {
  	//Do iteration
    //cout<<"a"<<endl;
  for (int n = 0; n < 41; n++)
  {
    ros::Time start = ros::Time::now();
    alg->updateFeatures();
    alg->expectationStep();
    alg->maximizationStep(applyEvidence);
    trackingVisualizer->update();
    scene.step(.03,2,.015);
    std::cout << (ros::Time::now() - start).toSec() << std::endl;
  }
  
  // ros::spinOnce();
  
  geometry_msgs::PoseArray vel_vec;

  geometry_msgs::Pose v_i;
  vel_vec.poses.resize(trackedObj->estVel_mio.size()); 
  // std::cout<<trackedObj->estVel_mio.at(0).getX()<<trackedObj->estVel_mio.at(0).getY()<<trackedObj->estVel_mio.at(0).getZ()<<std::endl;
  for (int k = 0; k < trackedObj->estVel_mio.size(); ++k){
      v_i.position.x=trackedObj->estVel_mio.at(k).getX();
      v_i.position.y=trackedObj->estVel_mio.at(k).getY();
      v_i.position.z=trackedObj->estVel_mio.at(k).getZ();
      v_i.orientation.w=1;
      v_i.orientation.x=0;
      v_i.orientation.y=0;
      v_i.orientation.z=0;
      vel_vec.poses[k]=v_i;
      // std::cout<<"x"<<trackedObj->estVel_mio.at(k).getX()<<std::endl;
      // std::cout<<"y"<<trackedObj->estVel_mio.at(k).getY()<<std::endl;
      // std::cout<<"z"<<trackedObj->estVel_mio.at(k).getZ()<<std::endl;
  }

  vel_pub.publish(vel_vec);

  objPub.publish(toTrackedObjectMessage(trackedObj));
  // toTrackedObjectMessage(trackedObj);
  // std::cout<<"a"<<std::)endl
  cloud_mia[0]=pc_vec[i];
  pc_Pub.publish(pc_vec[i-1]);
  image_mia[0]=imagevec[0];
  image_mia[1]=depthvec[0];
  callback(cloud_mia,image_mia);
  // rate.sleep();
    // }
  gesu_counter ++;
  // std::cout << "input: " << dio_counter << " output: " << gesu_counter << std::endl;
  }
  std::cout << "real input: " << gesu_counter << std::endl;
  
  // objPub.publish(toTrackedObjectMessage(trackedObj)); 
 


    

 	// }

}
