#include <ros/topic.h>
#include <ros/console.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "initialization.h"
#include "utils_tracking.h"
#include "config_tracking.h"
#include <geometry_msgs/Transform.h>
#include "simulation/config_bullet.h"
#include "utils/conversions.h"
#include <bulletsim_msgs/TrackedObject.h>
#include <bulletsim_msgs/Initialization.h>
#include <pcl/ros/conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include "tracked_object.h"
#include <tf/tf.h>
#include "simulation/bullet_io.h"
#include "simulation/softbodies.h"
#include "utils/logging.h"

//DEBUG
#include "physics_tracker.h"
#include "plotting_tracking.h"
#include "simulation/util.h"

#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <BulletSoftBody/btSoftBodyHelpers.h>

using namespace Eigen;

using namespace std;

TrackedObject::Ptr toTrackedObject(const bulletsim_msgs::ObjectInit& initMsg, ColorCloudPtr cloud, cv::Mat image, cv::Mat mask, CoordinateTransformer* transformer) {
  if (initMsg.type == "rope") {
	  vector<btVector3> nodes = toBulletVectors(initMsg.rope.nodes);
	  BOOST_FOREACH(btVector3& node, nodes) node += btVector3(0,0,.01);

	  CapsuleRope::Ptr sim(new CapsuleRope(scaleVecs(nodes,METERS), initMsg.rope.radius*METERS));
	  TrackedRope::Ptr tracked_rope(new TrackedRope(sim));

  	if (!image.empty())
  	  sim->setTexture(image, mask, toBulletTransform(transformer->camFromWorldEigen));

	  return tracked_rope;
  }
  else if (initMsg.type == "towel_corners") {

	  const vector<geometry_msgs::Point32>& points = initMsg.towel_corners.polygon.points;
	  vector<btVector3> corners = scaleVecs(toBulletVectors(points),METERS);

	  float sx = (corners[0] - corners[1]).length();
		float sy = (corners[0] - corners[3]).length();
		int resolution_x = sx/(TrackingConfig::node_distance*METERS) + 1;
		int resolution_y = sy/(TrackingConfig::node_distance*METERS) + 1;
		float mass = (TrackingConfig::surface_density/(METERS*METERS)) * (sx * sy);

	  printf("Created towel with following properties:\n");
	  printf("Surface density (Mass per area): %f\n", TrackingConfig::surface_density);
	  printf("Mass: %f\n", mass);
	  printf("Dimensions and area: %f x %f = %f\n", sx/METERS, sy/METERS, sx*sy/(METERS*METERS));
	  printf("Node distance (distance between nodes): %f\n", TrackingConfig::node_distance);
	  printf("Resolution: %d %d\n", resolution_x, resolution_y);

	  vector<btVector3> poly_corners = polyCorners(cloud);
	  //BOOST_FOREACH(btVector3& poly_corner, poly_corners) util::drawSpheres(poly_corner, Vector3f(1,0,0), 0.5, 2, env);
  	BulletSoftObject::Ptr sim = makeCloth(poly_corners, resolution_x, resolution_y, mass);
  	if (!image.empty())
  	  sim->setTexture(image, toBulletTransform(transformer->camFromWorldEigen));

	  //Shift the whole cloth upwards in case some of it starts below the table surface
	  sim->softBody->translate(btVector3(0,0,0.01*METERS));

	  //for (int i=0; i<sim->softBody->m_nodes.size(); i++) {
//		for (int i=0; i<10; i++) {
//			util::drawSpheres(sim->softBody->m_nodes[i].m_x, Vector3f(1,0,0), 0.5, 2, env);
//	  	cv::Point2f pixel = sim->getTexCoord(i);
//	  	image.at<cv::Vec3b>(pixel.y, pixel.x) = cv::Vec3b(255,255,255);
//	  }
//	  cv::imwrite("/home/alex/Desktop/tshirt_tex2.jpg", image);

	  TrackedCloth::Ptr tracked_towel(new TrackedCloth(sim, resolution_x, resolution_y, sx, sy));

	  return tracked_towel;
  }
  else if(initMsg.type=="custom_2d"){
                  float mass_insole=0.1;
                  btSoftBodyWorldInfo unusedWorldInfo;
                  btScalar*	vertices=new btScalar[339];
                  int triangles[531];
                  FILE *fd;
				  FILE *fp;
                  fd=fopen("/usr/ws/vertex.txt", "r"); 
                  fp=fopen("/usr/ws/triangle.txt", "r"); 
                  if( fd==NULL || fp==NULL) {
                       perror("Errore in apertura del file");
                       exit(1);
                     }
                  for(int i=0; i<339; i++)
                  {
                      fscanf(fd, "%f", &vertices[i]);
					  //cout<<vertices[i]<<endl;
					
                  }
				  for(int j=0;j<531;j++){
					fscanf(fp, "%i", &triangles[j]);
				  }
            
                  fclose(fd);
				  fclose(fp);
                  int ntriangles =177;
                  btSoftBody* psb=btSoftBodyHelpers::CreateFromTriMesh(unusedWorldInfo,vertices,triangles,ntriangles,true);
                  btSoftBody::Material* pm=psb->appendMaterial();
                  pm->m_kLST = 0.4;
                  psb->generateBendingConstraints(3,pm);
                  psb->setTotalMass(mass_insole);
                    
                  psb->generateClusters(16);
                  psb->getCollisionShape()->setMargin(0.001*METERS);
                    
                  psb->m_cfg.collisions	=	0;
                  //psb->m_cfg.collisions += btSoftBody::fCollision::SDF_RS; ///SDF based rigid vs soft
                  psb->m_cfg.collisions += btSoftBody::fCollision::CL_RS; ///Cluster vs convex rigid vs soft
                  //psb->m_cfg.collisions += btSoftBody::fCollision::VF_SS;	///Vertex vs face soft vs soft handling
                  psb->m_cfg.collisions += btSoftBody::fCollision::CL_SS; ///Cluster vs cluster soft vs soft handling
                  //psb->m_cfg.collisions	+= btSoftBody::fCollision::CL_SELF; ///Cluster soft body self collision                
                  psb->m_cfg.kDF = 1.0; // Dynamic friction coefficient
                  psb->m_cfg.piterations=1;
                  BulletSoftObject::Ptr sim = BulletSoftObject::Ptr(new BulletSoftObject(psb));
				  	if (!image.empty())
  	                   sim->setTexture(image, toBulletTransform(transformer->camFromWorldEigen));
                  
	                  //Shift the whole cloth upwards in case some of it starts below the table surface
	              sim->softBody->translate(btVector3(0,0,0.01*METERS));


	              TrackedCloth::Ptr tracked_insole(new TrackedCloth(sim,0,0,0,0));

	              return tracked_insole;

          }
  else if (initMsg.type == "box") {
		vector<btVector3> top_corners = polyCorners(cloud);
		float thickness = top_corners[0].z();
		BulletSoftObject::Ptr sim = makeSponge(top_corners, thickness, 3);
		sim->setColor(1,1,1,1);

	    //Shift the whole sponge upwards in case some of it starts below the table surface
	    sim->softBody->translate(btVector3(0,0,0.01*METERS));

		TrackedSponge::Ptr tracked_sponge(new TrackedSponge(sim));
		return tracked_sponge;
  }
  else
	  throw runtime_error("unrecognized initialization type" + initMsg.type);
}

bulletsim_msgs::TrackedObject toTrackedObjectMessage(TrackedObject::Ptr obj) {
  bulletsim_msgs::TrackedObject msg;
  msg.header.frame_id = "/ground";
  msg.header.stamp = ros::Time::now();

  if (obj->m_type == "rope") {
    msg.type = obj->m_type;
    msg.rope.nodes = toROSPoints(scaleVecs(obj->getPoints(), 1/METERS));
  }
  else if (obj->m_type == "towel"){
  	msg.type = obj->m_type;

  	BulletSoftObject::Ptr sim = boost::dynamic_pointer_cast<BulletSoftObject>(obj->m_sim);
  	const btSoftBody::tNodeArray& nodes = sim->softBody->m_nodes;
  	const btSoftBody::tFaceArray& faces = sim->softBody->m_faces;

  	for (int i=0; i<nodes.size(); i++) {
  		msg.mesh.vertices.push_back(toROSPoint(nodes[i].m_x/METERS));
  		msg.mesh.normals.push_back(toROSPoint(nodes[i].m_n/METERS));
  	}

  	// compute face to nodes indices
  	vector<vector<int> > face2nodes(faces.size(), vector<int>(3,-1));
  	for (int i=0; i<nodes.size(); i++) {
  		int j,c;
  		for(j=0; j<faces.size(); j++) {
  			for(c=0; c<3; c++) {
  				if (&nodes[i] == faces[j].m_n[c]) {
  					face2nodes[j][c] = i;
  				}
  			}
  		}
  	}

  	for (int j=0; j<faces.size(); j++) {
  		bulletsim_msgs::Face face;
  		for (int c=0; c<3; c++) {
  			face.vertex_inds.push_back(face2nodes[j][c]);
  			face.normal_inds.push_back(face2nodes[j][c]);
  		}
  		msg.mesh.faces.push_back(face);
  	}

  }
  else {
	  //TODO
	  //LOG_ERROR("I don't knot how to publish a ");
  }
  return msg;
}

TrackedObject::Ptr callInitServiceAndCreateObject(ColorCloudPtr cloud, cv::Mat image, cv::Mat mask, CoordinateTransformer* transformer) {
  bulletsim_msgs::Initialization init;
  pcl::toROSMsg(*scaleCloud(cloud, 1/METERS), init.request.cloud);
  init.request.cloud.header.frame_id = "/ground";
  init.response.objectInit.type="custom_2d";
   	
//   bool success = ros::service::call(initializationService, init);
  bool success=true;
  if (success)
  	return toTrackedObject(init.response.objectInit, cloud, image, mask, transformer);
  else {
		ROS_ERROR("initialization failed");
		return TrackedObject::Ptr();
  }
}


