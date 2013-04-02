//#include "Model_Builder.h"
#include "features.h"

float sift_ratio(0.7);
float sampling_rate(7.0);
float CSHOT_threshold(0.25);


void GetCSHOTCorrs(pcl::PointCloud<Descriptor3DType>::Ptr model_descriptors, pcl::PointCloud<Descriptor3DType>::Ptr scene_descriptors, pcl::CorrespondencesPtr model_scene_corrs)
{
	int i,j, modelNum = model_descriptors->size(), sceneNum = scene_descriptors->size();

	for( i = 0 ; i < modelNum ; i++ ){
		float dist_max = 0;
		int idx_max = -1;
		for( j = 0 ; j < sceneNum; j++ )
		{
			float dist = SimHist(model_descriptors->points[i].descriptor, scene_descriptors->points[j].descriptor, 1344);
			if( dist > dist_max )
			{
				dist_max = dist;
				idx_max = j;
			}
		}
		std::cerr<<dist_max<<std::endl;
		if( dist_max > 0.8 )
		{
			pcl::Correspondence corr (i, idx_max, dist_max);
			model_scene_corrs->push_back (corr);
		}
	}
	/*
	pcl::KdTreeFLANN<Descriptor3DType> match_search;
	match_search.setInputCloud (scene_descriptors);

	//  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
	for (size_t i = 0; i < model_descriptors->size (); ++i)
	{
		std::vector<int> neigh_indices (1);
		std::vector<float> neigh_sqr_dists (1);
		if (!pcl_isfinite (model_descriptors->at(i).descriptor[0])) //skipping NaNs
		{
			std::cerr<<"Hi"<<std::endl;
			continue;
		}

		int found_neighs = match_search.nearestKSearch (model_descriptors->at (i), 2, neigh_indices, neigh_sqr_dists);
		std::cerr<<neigh_sqr_dists[0]<<std::endl;
		if(found_neighs >= 1 && neigh_sqr_dists[0] < CSHOT_threshold) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
		{
			//std::cerr<<neigh_sqr_dists[0]<<std::endl;
			pcl::Correspondence corr (static_cast<int> (i), neigh_indices[0], neigh_sqr_dists[0]);
			model_scene_corrs->push_back (corr);
		}
	}
	*/
	std::cout << "CSHOT Correspondences found: " << model_scene_corrs->size () << std::endl;
}

int main (int argc, char *argv[])
{     
	int s_idx, num, i, j, k;
	std::string modelname(argv[1]);  
	std::string scenename(argv[2]); 

	float ratio1=25, ratio2=25;

	pcl::console::parse_argument (argc, argv, "--ratio1", ratio1);
	pcl::console::parse_argument (argc, argv, "--ratio2", ratio2);
	pcl::console::parse_argument (argc, argv, "--CSHOT_ratio", CSHOT_ratio);
	pcl::console::parse_argument (argc, argv, "--CSHOT_threshold", CSHOT_threshold);

	std::cerr<<sampling_rate<<" "<<CSHOT_ratio<<" "<<CSHOT_threshold<<std::endl;

	pcl::PointCloud<PointXYZRGBIM>::Ptr model(new pcl::PointCloud<PointXYZRGBIM>);
	pcl::PointCloud<PointT>::Ptr model_rgb (new pcl::PointCloud<PointT>());
	pcl::io::loadPCDFile(modelname, *model);
	pcl::copyPointCloud(*model, *model_rgb);
	
	float resolution = static_cast<float>(computeCloudResolution(model_rgb));
	std::cerr<<"resolution: "<<resolution<<std::endl;

	pcl::PointCloud<NormalT>::Ptr model_normals(new pcl::PointCloud<NormalT>);
	computeNormals(model_rgb, model_normals, resolution*normal_ratio);

	cv::Mat model_image = cv::Mat::zeros(480, 640, CV_8UC1);
	cv::Mat model_2DTo3D = cv::Mat::zeros(480,640, CV_32SC1);
	std::vector<cv::KeyPoint> model_sift_keypoints;
	cv::Mat old_model_descr;
	if (ComputeSIFT(model, model_image, model_2DTo3D, model_sift_keypoints, old_model_descr) == false)
	{
		std::cerr<<"Failed to Compute SIFT Features"<<std::endl;
		return 0;
	}
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_temp_sift(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	cv::Mat model_descr;
	GetRawSIFT(model, model_normals, model_sift_keypoints, model_2DTo3D, old_model_descr, model_temp_sift, model_descr);
	pcl::PointCloud<PointT>::Ptr model_keypoints(new pcl::PointCloud<PointT>);
	pcl::copyPointCloud(*model_temp_sift, *model_keypoints);

	pcl::SHOTColorEstimationOMP<PointT, NormalT, Descriptor3DType, pcl::ReferenceFrame> descr_est;
	descr_est.setRadiusSearch (resolution*ratio1);
	descr_est.setLRFRadius (resolution*ratio2);
	descr_est.setNumberOfThreads(8);
	pcl::PointCloud<Descriptor3DType>::Ptr model_CSHOTDescr(new pcl::PointCloud<Descriptor3DType>());
	descr_est.setInputCloud (model_keypoints);
	descr_est.setInputNormals (model_normals);
	descr_est.setSearchSurface (model_rgb);
	descr_est.compute (*model_CSHOTDescr);

	pcl::PointCloud<PointXYZRGBIM>::Ptr scene(new pcl::PointCloud<PointXYZRGBIM>);
	pcl::PointCloud<PointT>::Ptr scene_rgb (new pcl::PointCloud<PointT>());
	pcl::io::loadPCDFile(scenename, *scene);
	pcl::copyPointCloud(*scene, *scene_rgb);

	pcl::PointCloud<NormalT>::Ptr scene_normals(new pcl::PointCloud<NormalT>);
	computeNormals(scene_rgb, scene_normals, resolution*normal_ratio);

	std::cerr<<"Hi"<<std::endl;

	cv::Mat scene_image = cv::Mat::zeros(480, 640, CV_8UC1);
	cv::Mat scene_2DTo3D = cv::Mat::zeros(480,640, CV_32SC1);
	std::vector<cv::KeyPoint> scene_sift_keypoints;
	cv::Mat old_scene_descr;
	if (ComputeSIFT(scene, scene_image, scene_2DTo3D, scene_sift_keypoints, old_scene_descr) == false)
	{
		std::cerr<<"Failed to Compute SIFT Features"<<std::endl;
		return 0;
	}
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene_temp_sift(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	cv::Mat scene_descr;
	GetRawSIFT(scene, scene_normals, scene_sift_keypoints, scene_2DTo3D, old_scene_descr, scene_temp_sift, scene_descr);
	pcl::PointCloud<PointT>::Ptr scene_keypoints(new pcl::PointCloud<PointT>);
	pcl::copyPointCloud(*scene_temp_sift, *scene_keypoints);

	pcl::SHOTColorEstimationOMP<PointT, NormalT, Descriptor3DType, pcl::ReferenceFrame> descr_est1;
	descr_est1.setRadiusSearch (resolution*ratio1);
	descr_est1.setLRFRadius (resolution*ratio2);
	descr_est1.setNumberOfThreads(8);
	pcl::PointCloud<Descriptor3DType>::Ptr scene_CSHOTDescr(new pcl::PointCloud<Descriptor3DType>());
	descr_est1.setInputCloud (scene_keypoints);
	descr_est1.setInputNormals (scene_normals);
	descr_est1.setSearchSurface (scene_rgb);
	descr_est1.compute (*scene_CSHOTDescr);
	
	std::cerr<<"Model Keypoints: "<<model_keypoints->size()<<" Scene Keypoints: "<<scene_keypoints->size()<<std::endl;

	pcl::CorrespondencesPtr model_scene_corrs( new pcl::Correspondences ());
	GetCSHOTCorrs(model_CSHOTDescr, scene_CSHOTDescr, model_scene_corrs);

	std::cerr<<"Correspondences Number: "<<model_scene_corrs->size()<<std::endl;
	
	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer("Aligned Model"));
	viewer->initCameraParameters();

	viewer->addPointCloud( scene_rgb, "scene");
	pcl::PointCloud<PointT>::Ptr off_model_rgb (new pcl::PointCloud<PointT>());
	pcl::PointCloud<PointT>::Ptr off_model_keypoints_rgb (new pcl::PointCloud<PointT>());
	pcl::transformPointCloud (*model_rgb, *off_model_rgb, Eigen::Vector3f (-0.5,0,0), Eigen::Quaternionf (1, 0, 0, 0));
	pcl::transformPointCloud (*model_keypoints, *off_model_keypoints_rgb, Eigen::Vector3f (-0.5,0,0), Eigen::Quaternionf (1, 0, 0, 0));
	
	viewer->addPointCloud( off_model_rgb, "model");
	
	for(int i = 0 ; i < model_scene_corrs->size() ; i++ )
	{
		std::ostringstream convert;			// stream used for the conversion
		convert << i;

		PointT temp1 = off_model_keypoints_rgb->points[model_scene_corrs->at(i).index_query];
		PointT temp2 = scene_keypoints->points[model_scene_corrs->at(i).index_match];

		viewer->addLine(temp1, temp2, 0, 255, 0, "Line "+ convert.str());
		viewer->spin();
		viewer->removeAllShapes();
	}
	
	return 1;
}

	/*
	for( int i = 0 ; i < model_keypoints->size(); i++ )
	{
		PointT temp = model_keypoints->points[i];
		pcl::search::KdTree<PointT> tree;
		tree.setInputCloud(model_rgb);

		std::vector<int> neigh_indices;
		std::vector<float> neigh_sqr_dists;
		tree.radiusSearch(temp, resolution*CSHOT_ratio, neigh_indices, neigh_sqr_dists);
		
		pcl::PointCloud<PointT>::Ptr tempCloud(new pcl::PointCloud<PointT>());
		for(int j = 0 ; j < neigh_indices.size() ; j++ )
			tempCloud->points.push_back(model_rgb->points[neigh_indices[j]]);

		pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer("Model"));
		viewer->initCameraParameters();
		viewer->addPointCloud(model_rgb, "model");

		pcl::visualization::PointCloudColorHandlerCustom<PointT> single_color(tempCloud, 255, 0, 0);
		viewer->addPointCloud<PointT> (tempCloud, single_color, "tempRegion");
		viewer->spin();
	}
	*/


/*
void GetCSHOTCorrs(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene, float model_resolution)
{
	pcl::PointCloud<int> sampled_indices;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_keypoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene_keypoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

	pcl::UniformSampling<pcl::PointXYZRGBNormal> uniform_sampling;
	uniform_sampling.setInputCloud (model);
	uniform_sampling.setRadiusSearch (model_resolution*sampling_rate);
	uniform_sampling.compute (sampled_indices);
	pcl::copyPointCloud (*model, sampled_indices.points, *model_keypoints);
	std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

	uniform_sampling.setInputCloud (scene);
	uniform_sampling.setRadiusSearch (model_resolution*sampling_rate);
	sampled_indices.clear();
	uniform_sampling.compute(sampled_indices);
	pcl::copyPointCloud (*scene, sampled_indices.points, *scene_keypoints);
	std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;

	pcl::PointCloud<PointT>::Ptr model_rgb (new pcl::PointCloud<PointT>());
	pcl::PointCloud<PointT>::Ptr scene_rgb (new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr model_normals (new pcl::PointCloud<NormalT>());
	pcl::PointCloud<NormalT>::Ptr scene_normals (new pcl::PointCloud<NormalT>());
	pcl::copyPointCloud (*model, *model_rgb);
	pcl::copyPointCloud (*model, *model_normals);
	pcl::copyPointCloud (*scene, *scene_rgb);
	pcl::copyPointCloud (*scene, *scene_normals);
	pcl::PointCloud<PointT>::Ptr model_keypoints_rgb (new pcl::PointCloud<PointT>());
	pcl::PointCloud<PointT>::Ptr scene_keypoints_rgb (new pcl::PointCloud<PointT>());
	pcl::copyPointCloud (*model_keypoints, *model_keypoints_rgb);
	pcl::copyPointCloud (*scene_keypoints, *scene_keypoints_rgb);

	pcl::PointCloud<Descriptor3DType>::Ptr model_descriptors (new pcl::PointCloud<Descriptor3DType> ());
	pcl::PointCloud<Descriptor3DType>::Ptr scene_descriptors (new pcl::PointCloud<Descriptor3DType> ());

	pcl::SHOTColorEstimationOMP<PointT, NormalT, Descriptor3DType, pcl::ReferenceFrame> descr_est;
	//pcl::SHOTEstimationOMP<PointT, NormalT, pcl::SHOT352> descr_est;
	descr_est.setRadiusSearch (model_resolution*CSHOT_ratio);
	descr_est.setNumberOfThreads(8);

	descr_est.setInputCloud (model_keypoints_rgb);
	descr_est.setInputNormals (model_normals);
	descr_est.setSearchSurface (model_rgb);
	descr_est.compute (*model_descriptors);

	descr_est.setInputCloud (scene_keypoints_rgb);
	descr_est.setInputNormals (scene_normals);
	descr_est.setSearchSurface (scene_rgb);
	descr_est.compute (*scene_descriptors);

	//
	//  Find Model-Scene Correspondences with KdTree
	//
	pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

	pcl::KdTreeFLANN<Descriptor3DType> match_search;
	match_search.setInputCloud (model_descriptors);

	//  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
	for (size_t i = 0; i < scene_descriptors->size (); ++i)
	{
		std::vector<int> neigh_indices (1);
		std::vector<float> neigh_sqr_dists (1);
		if (!pcl_isfinite (scene_descriptors->at(i).descriptor[0])) //skipping NaNs
			continue;

		int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
		if(found_neighs == 1 && neigh_sqr_dists[0] < CSHOT_threshold) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
		{
			pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
			model_scene_corrs->push_back (corr);
		}
	}
	std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;
	
	if( true )
	{
		pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer("Aligned Model"));
		viewer->initCameraParameters();

		viewer->addPointCloud( scene_rgb, "scene");
		pcl::PointCloud<PointT>::Ptr off_model_rgb (new pcl::PointCloud<PointT>());
		pcl::PointCloud<PointT>::Ptr off_model_keypoints_rgb (new pcl::PointCloud<PointT>());
		pcl::transformPointCloud (*model_rgb, *off_model_rgb, Eigen::Vector3f (-0.5,0,0), Eigen::Quaternionf (1, 0, 0, 0));
		pcl::transformPointCloud (*model_keypoints_rgb, *off_model_keypoints_rgb, Eigen::Vector3f (-0.5,0,0), Eigen::Quaternionf (1, 0, 0, 0));
	
		viewer->addPointCloud( off_model_rgb, "model");
	
		for(int i = 0 ; i < model_scene_corrs->size() ; i++ )
		{
			std::ostringstream convert;			// stream used for the conversion
			convert << i;

			PointT temp1 = off_model_keypoints_rgb->points[model_scene_corrs->at(i).index_query];
			PointT temp2 = scene_keypoints_rgb->points[model_scene_corrs->at(i).index_match];

			viewer->addLine(temp1, temp2, 0, 255, 0, "Line "+ convert.str());
			viewer->spin();
		}
		
	}
	
}
*/
/*
int main (int argc, char *argv[])
{     
	int s_idx, num, i, j, k;
	std::string inputpath(argv[1]);                 //path for raw scene data  
	std::string model_name("Model_");
	std::string postfix("_c.pcd");

	if( sscanf( argv[2], "%d", &s_idx ) != 1 ){   //starting index (integer)
        std::cerr<<"Please input m_spcount\n";
        return -1;
    }
    if( sscanf( argv[3], "%d", &num ) != 1 ){   //ending index (integer)
        std::cerr<<"Please input m_spcount\n";
        return -1;
    }

	pcl::console::parse_argument (argc, argv, "--sampling_rate", sampling_rate);
	pcl::console::parse_argument (argc, argv, "--CSHOT_ratio", CSHOT_ratio);
	pcl::console::parse_argument (argc, argv, "--CSHOT_threshold", CSHOT_threshold);

	std::cerr<<sampling_rate<<" "<<CSHOT_ratio<<" "<<CSHOT_threshold<<std::endl;
	//Model_Builder cshot;

	for( i = s_idx ; i < s_idx + num - 1; i++ )
	{
		std::ostringstream convert1, convert2;			// stream used for the conversion
		convert1 << i + 1;
		convert2 << i;

		std::string cloudname1 = inputpath + "\\Model_" + convert1.str() + postfix;
		std::cerr<<cloudname1<<std::endl;
		pcl::PointCloud<PointXYZRGBIM>::Ptr model(new pcl::PointCloud<PointXYZRGBIM>);
		pcl::PointCloud<PointT>::Ptr model_rgb (new pcl::PointCloud<PointT>());
		pcl::io::loadPCDFile(cloudname1, *model);
		pcl::copyPointCloud(*model, *model_rgb);

		float model_resolution = static_cast<float>(computeCloudResolution(model_rgb));

		pcl::PointCloud<pcl::Normal>::Ptr model_normals (new pcl::PointCloud<pcl::Normal>);
		pcl::io::loadPCDFile(inputpath + "\\Model_normals_" + convert1.str() + postfix, *model_normals);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_rgb_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
		pcl::copyPointCloud(*model_rgb, *model_rgb_normals);
		pcl::copyPointCloud(*model_normals, *model_rgb_normals);

		std::string cloudname2 = inputpath + "\\Model_" + convert2.str() + postfix;
		std::cerr<<cloudname2<<std::endl;
		pcl::PointCloud<PointXYZRGBIM>::Ptr scene(new pcl::PointCloud<PointXYZRGBIM>);
		pcl::io::loadPCDFile(cloudname2, *scene);
		pcl::PointCloud<PointT>::Ptr scene_rgb (new pcl::PointCloud<PointT>());
		pcl::copyPointCloud(*scene, *scene_rgb);

		pcl::PointCloud<pcl::Normal>::Ptr scene_normals (new pcl::PointCloud<pcl::Normal>);
		pcl::io::loadPCDFile(inputpath + "\\Model_normals_" + convert1.str() + postfix, *scene_normals);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene_rgb_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
		pcl::copyPointCloud(*scene_rgb, *scene_rgb_normals);
		pcl::copyPointCloud(*scene_normals, *scene_rgb_normals);

		GetCSHOTCorrs(model_rgb_normals, scene_rgb_normals, model_resolution);

	}

    return (0);
}
*/