#include "Model_Builder.h"

float sift_ratio(0.6);
float sampling_rate(10), CSHOT_threshold(0.25), ransac_threshold(0.01);

std::string path;
//10 5
//#define BSTEP 5

//float planeCoef[4] = {0.0414733, -0.598233, 0.800248, -0.508407};	//ied

//float planeCoef[4] = {0.043893, -0.906755, 0.419368, -0.506173};	   //3

//float planeCoef[4] = { 0.0216105, -0.911367, 0.411027, -0.483404 };  //2

//float planeCoef[4] = { 0.027479, -0.88449, 0.46575, -0.433954 };  //dense

//float planeCoef[4] = {0.00254148, -0.662382, 0.749162, -0.521348};	//drill

//float planeCoef[4] = {0.0326326, -0.857815, 0.512921, -0.548568};	//shell

//float planeCoef[4] = {-0.0158295, -0.608154, 0.793661, -0.529778};	//screwdriver

//float planeCoef[4] = {-0.0237031, -0.879364, 0.475559, -0.548501};	//extinguisher

//float planeCoef[4] = {0.00536737, -0.619923, 0.784644, -0.518848};	//grenade

//float planeCoef[4] = {-0.0255116, -0.547279, 0.836561, -0.528179};	//c4

float planeCoef[4] = {0.00118571, -0.64546, 0.763793, -0.49045};	//drill2

void ShowFeatureTransform(pcl::PointCloud<PointT>::Ptr model, pcl::PointCloud<PointT>::Ptr scene, 
						  pcl::PointCloud<PointT>::Ptr model_keypoints, pcl::PointCloud<PointT>::Ptr scene_keypoints, 
						  pcl::Correspondences &corrs, Eigen::Matrix4f &rotatranslation);

int main(int argc, char **argv)
{
	int index_s,index_e, i, j, num, s_idx, midpoint=0, count=0, keyframe_inliers = 200, BLen = 7;
	float cropHeight = 0.04;
    path = argv[1];										//path for raw scene data       
	std::string modelname;
	if( sscanf( argv[2], "%d", &s_idx ) != 1 ){			//starting index (integer)
        std::cerr<<"Please input m_spcount\n";
        return 0;
	}
	if( sscanf( argv[3], "%d", &num ) != 1 ){   //starting index (integer)
        std::cerr<<"Please input m_spcount\n";
        return 0;
	}
	bool show_keypoints = false, show_shift = false, filesaved = false, show_object = false, show_region = false; 

	if (pcl::console::find_switch (argc, argv, "-sk"))
		show_keypoints = true;
	if (pcl::console::find_switch (argc, argv, "-ss"))
		show_shift = true;
	if (pcl::console::find_switch (argc, argv, "-so"))
		show_object = true;
	if (pcl::console::find_switch (argc, argv, "-sr"))
		show_region = true;

	pcl::console::parse_argument (argc, argv, "--sift_ratio", sift_ratio);
	pcl::console::parse_argument (argc, argv, "--cropHeight", cropHeight);
	pcl::console::parse_argument (argc, argv, "--keyframe_inliers", keyframe_inliers);
	pcl::console::parse_argument (argc, argv, "--BLen", BLen);
	pcl::console::parse_argument (argc, argv, "--sampling_rate", sampling_rate);
	pcl::console::parse_argument (argc, argv, "--CSHOT_ratio", CSHOT_ratio);
	pcl::console::parse_argument (argc, argv, "--CSHOT_threshold", CSHOT_threshold);
	pcl::console::parse_argument (argc, argv, "--ransac_threshold", ransac_threshold);

	if( pcl::console::parse_argument (argc, argv, "--filesaved", modelname) >= 0 )
		filesaved = true;

	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer("Aligned Model"));
	viewer->initCameraParameters();

	pcl::PointCloud<pcl::PointNormal>::Ptr camera_pos(new pcl::PointCloud<pcl::PointNormal>());
	pcl::PointNormal camera_ori;
	camera_ori.x = 0;camera_ori.y = 0;camera_ori.z = 0;camera_ori.normal_x = 0;camera_ori.normal_y = 0;camera_ori.normal_z = 1;
	camera_pos->points.push_back(camera_ori);
	
	//Initialize ModelBuilder
	Model_Builder ModelBuilder;
	ModelBuilder.setICPThreshold(0.07);
	ModelBuilder.setRANSACThreshold(ransac_threshold);
	ModelBuilder.setSIFTRatio(sift_ratio);
	ModelBuilder.setKeyframeInliers(keyframe_inliers);
	ModelBuilder.setNormalRatio(normal_ratio);
	ModelBuilder.setRFRatio(normal_ratio);
	ModelBuilder.setPlaneCoef(planeCoef);
	ModelBuilder.setBLen(BLen);
	ModelBuilder.setSamplingRate(sampling_rate);
	ModelBuilder.setCSHOTRatio(CSHOT_ratio);
	ModelBuilder.setCSHOTThreshold(CSHOT_threshold);
	
	ModelBuilder.setShowKeypoints(show_keypoints);
	ModelBuilder.setShowShift(show_shift);
	ModelBuilder.setShowRegion(show_region);
	ModelBuilder.setShowObject(show_object);
	ModelBuilder.setcropHeight(cropHeight);

	ModelBuilder.setViewer(viewer);

	for( i = s_idx; i < s_idx + num + 1 ; i++ )
	{   
		int id;
		if( i == s_idx + num )
			id = i -1;
		else
			id = i;
		std::ostringstream convert;			// stream used for the conversion
		convert << id;
		
		pcl::PointCloud<PointT>::Ptr Filtered_region(new pcl::PointCloud<PointT>);
		pcl::PointCloud<NormalT>::Ptr Filtered_region_normals (new pcl::PointCloud<NormalT>);

		std::string cloudname = path + "\\Model_" + convert.str() + "_c.pcd";
		std::cerr<<cloudname<<std::endl;
		pcl::PointCloud<PointXYZRGBIM>::Ptr model(new pcl::PointCloud<PointXYZRGBIM>);
		pcl::io::loadPCDFile(cloudname, *model);

		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_rgb_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
		pcl::io::loadPCDFile(path + "\\Model_" + convert.str() + "_g.pcd", *model_rgb_normals);

		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_keypoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
		pcl::io::loadPCDFile(path + "\\Model_keypoints_" + convert.str() + "_g.pcd", *model_keypoints);
		
		pcl::PointCloud<SIFTDescr>::Ptr modelSIFTDescr(new pcl::PointCloud<SIFTDescr>());
		pcl::io::loadPCDFile(path + "\\Model_sift_" + convert.str() + "_g.pcd", *modelSIFTDescr);

		pcl::PointCloud<Descriptor3DType>::Ptr modelCSHOTDescr(new pcl::PointCloud<Descriptor3DType>());
		pcl::io::loadPCDFile(path + "\\Model_cshot_" + convert.str() + "_g.pcd", *modelCSHOTDescr);

		ModelBuilder.Process(model, model_rgb_normals, model_keypoints, modelSIFTDescr, modelCSHOTDescr);
		
	}
	std::cerr<<"**********Generating Full Model*****************"<<std::endl;
	ModelBuilder.GenModel();
	if( filesaved == true)
		ModelBuilder.SaveModel(modelname);

	viewer->spin();
	
	return 1;
}

void ShowFeatureTransform(pcl::PointCloud<PointT>::Ptr model, pcl::PointCloud<PointT>::Ptr scene, 
						  pcl::PointCloud<PointT>::Ptr model_keypoints, pcl::PointCloud<PointT>::Ptr scene_keypoints, 
						  pcl::Correspondences &corrs, Eigen::Matrix4f &rotatranslation)
{
	int corrs_num = corrs.size();
	pcl::PointCloud<PointT>::Ptr model_inliers(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr buffer(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr scene_inliers(new pcl::PointCloud<PointT>);
	for(size_t i = 0 ; i < corrs_num ; i++)
	{
		//std::cerr<<corrs.at(i).index_query<<" "<<corrs.at(i).index_match<<std::endl;
		model_inliers->points.push_back(model_keypoints->points[corrs.at(i).index_query]);
		scene_inliers->points.push_back(scene_keypoints->points[corrs.at(i).index_match]);
	}
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Coarse Alignment"));
	viewer->initCameraParameters();
	viewer->addPointCloud(scene, "scene");
	pcl::PointCloud<PointT>::Ptr temp(new pcl::PointCloud<PointT>);
	pcl::transformPointCloud(*model, *temp, rotatranslation);
	pcl::transformPointCloud(*model_inliers, *buffer, rotatranslation);
	pcl::copyPointCloud(*buffer, *model_inliers);
	pcl::visualization::PointCloudColorHandlerCustom<PointT> model_color_handler (temp, 255, 0, 0);
	viewer->addPointCloud(temp, model_color_handler, "model");

	pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_inliers_handler (scene_inliers, 0, 0, 255);
    viewer->addPointCloud (scene_inliers, scene_inliers_handler, "scene_inliers");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_inliers");

	pcl::visualization::PointCloudColorHandlerCustom<PointT> model_inliers_handler (model_inliers, 0, 255, 0);
    viewer->addPointCloud (model_inliers, model_inliers_handler, "model_inliers");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "model_inliers");
	viewer->spin();
	
}

/*
		cv::Mat model_image;
		cv::Mat model_2DTo3D;
		std::vector<cv::KeyPoint> model_sift_keypoints;
		cv::Mat model_matrix_sift_keypoints;
		cv::Mat pre_sift_descriptors;
		cv::FileStorage fs1(path + "\\Model_sift_" + convert.str() + "_c.xml", cv::FileStorage::READ);
		fs1["model_image"] >> model_image;	
		fs1["model_2DTo3D"] >> model_2DTo3D;
		fs1["model_sift_keypoints"] >> model_matrix_sift_keypoints;
		fs1["model_sift_descriptors"] >> pre_sift_descriptors;
		for( j = 0 ; j < model_matrix_sift_keypoints.rows; j++ )
		{
			cv::KeyPoint temp;
			temp.pt.y = model_matrix_sift_keypoints.at<float>(j, 0);
			temp.pt.x = model_matrix_sift_keypoints.at<float>(j, 1);
			temp.octave = model_matrix_sift_keypoints.at<float>(j, 2);
			temp.angle = model_matrix_sift_keypoints.at<float>(j, 3);
			model_sift_keypoints.push_back(temp);
		}
		//std::cerr<<"SIFT keypoints: "<<model_sift_keypoints.size()<<" "<<pre_sift_descriptors.rows<<std::endl;

		
		cv::Mat model_sift_descriptors;
		ModelBuilder.GetRawSIFT(model, model_normals, model_sift_keypoints, model_2DTo3D, pre_sift_descriptors, model_keypoints, model_sift_descriptors);
	
*/

/*
	pcl::PointCloud<pcl::PointXYZ>::Ptr show_ori_pos(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<NormalT>::Ptr show_ori_dir(new pcl::PointCloud<NormalT>);
	pcl::PointXYZ my_ori_pos(0,0,0);
	NormalT my_ori_dir(0,0,1);
	show_ori_pos->points.push_back(my_ori_pos);
	show_ori_dir->points.push_back(my_ori_dir);
	pcl::PointCloud<pcl::PointXYZ>::Ptr show_camera_pos(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<NormalT>::Ptr show_camera_dir(new pcl::PointCloud<NormalT>);
	pcl::copyPointCloud(*camera_pos, *show_camera_pos);
	pcl::copyPointCloud(*camera_pos, *show_camera_dir);
	viewer.addPointCloudNormals<pcl::PointXYZ, NormalT>(show_camera_pos, show_camera_dir, 1, 0.1, "camera_pos");
	viewer.addPointCloudNormals<pcl::PointXYZ, NormalT>(show_ori_pos, show_ori_dir, 1, 0.1, "camera_ori");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255, 0, 0, "camera_ori");
	*/
	/*
	pcl::PointCloud<PointT>::Ptr reduced_model_rgb (new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr reduced_model_normals (new pcl::PointCloud<NormalT>());
	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(showmodel);
	sor.setLeafSize(model_resolution, model_resolution, model_resolution);
	sor.filter(*reduced_model_rgb);
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_shift(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr reduced_model_center(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr model_rf(new pcl::PointCloud<pcl::ReferenceFrame> ());
	
	computeNormals(reduced_model_rgb, reduced_model_normals, model_resolution*NORMAL_RATIO);
	ComputeCentroid(reduced_model_rgb, reduced_model_center);
	AdjustNormals(reduced_model_rgb, reduced_model_normals, reduced_model_center->points[0]);
	ComputeCloudRF(showSIFT, reduced_model_normals, reduced_model_rgb, model_rf, model_resolution*RF_RATIO);
	ComputeShift(showSIFT, model_rf, model_shift, reduced_model_center);
	
	computeKeyNormals(showSIFT, showSIFT_normals, reduced_model_rgb, model_resolution*NORMAL_RATIO);
	AdjustNormals(showSIFT, showSIFT_normals, reduced_model_center->points[0]);
	
	pcl::PointCloud<PointObj>::Ptr model_keypoints_final(new pcl::PointCloud<PointObj>);
	pcl::copyPointCloud(*showSIFT, *model_keypoints_final);
	pcl::copyPointCloud(*showSIFT_normals, *model_keypoints_final);
	
	for(int k=0 ; k<model_keypoints_final->points.size() ; k++){
		memcpy(model_keypoints_final->at(k).rf, model_rf->at(k).rf, sizeof(float)*9);
		memcpy(model_keypoints_final->at(k).data_s, model_shift->at(k).data, sizeof(float)*4);
	}

	pcl::io::savePCDFile(saved_prefix+"_siftkeypoints_f.pcd", *model_keypoints_final, true);*/