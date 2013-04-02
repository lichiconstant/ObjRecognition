#include "Model_Builder.h"

Model_Builder::Model_Builder() : show_object(false), show_keypoints(false), show_shift(false), show_region(false)
{
	icp_inlier_threshold = 0.07;
	ransac_inlier_threshold = 0.01;
	sift_ratio = 0.6;
	rf_ratio = 25;
	normal_ratio = 25;
	keyframe_inliers = 40;

	pre_initial_guess = Eigen::Matrix4f::Identity();

	pcl::PointCloud<PointObj>::Ptr fullmodel_(new pcl::PointCloud<PointObj>());
	pcl::PointCloud<PointObj>::Ptr fullmodel_SIFT_(new pcl::PointCloud<PointObj>());
	pcl::PointCloud<PointObj>::Ptr truemodel_(new pcl::PointCloud<PointObj>());
	pcl::PointCloud<PointObj>::Ptr trueSIFT_(new pcl::PointCloud<PointObj>());
	pcl::PointCloud<PointObj>::Ptr refined_trueSIFT_(new pcl::PointCloud<PointObj>);
	pcl::PointCloud<SIFTDescr>::Ptr final_sift_(new pcl::PointCloud<SIFTDescr>());
	pcl::PointCloud<Descriptor3DType>::Ptr final_cshot_(new pcl::PointCloud<Descriptor3DType>());

	pcl::PointCloud<SIFTDescr>::Ptr SIFTDescr_pool_(new pcl::PointCloud<SIFTDescr>());	//raw SIFT descriptors
	pcl::PointCloud<Descriptor3DType>::Ptr CSHOTDescr_pool_(new pcl::PointCloud<Descriptor3DType>());	//raw CSHOT descriptors
	
	fullmodel = fullmodel_;
	fullmodel_SIFT = fullmodel_SIFT_;
	truemodel = truemodel_;
	trueSIFT = trueSIFT_;
	SIFTDescr_pool  = SIFTDescr_pool_;
	CSHOTDescr_pool = CSHOTDescr_pool_;
	refined_trueSIFT = refined_trueSIFT_;
	final_cshot = final_cshot_;
	final_sift = final_sift_;
	
	BLen = 10;

	sampling_rate = 10.0;
	CSHOT_ratio = 30;
	CSHOT_threshold = 0.25f;

	cropHeight = 0.04;
}

Model_Builder::~Model_Builder()
{
}

void Model_Builder::Find_SIFT_Corrs(pcl::PointCloud<SIFTDescr>::Ptr descr1, pcl::PointCloud<SIFTDescr>::Ptr descr2, pcl::CorrespondencesPtr model_scene_corrs, bool *flag1, bool *flag2)
{
	//Find best correspondences is descr2 for descr1
	int num1 = descr1->points.size();
	int num2 = descr2->points.size();
	int i, j, best_idx;
	float *sim_score = new float[num2];
	memset(sim_score, 0, sizeof(float)*num2);
	double best_score, match_score, second_best_score;
	for( i = 0 ; i < num1; i++ )
	{
		if( flag1[i] == false )
		{
			best_score = 100000;
			second_best_score = best_score+1;
			for( j=0 ; j<num2 ; j++ )
			{
				if( flag2[j] == false)
				{
					//match_score = acos( descr1.row(i).dot(descr2.row(j)));
					float dotProduct = 0;
					cv::Mat d1 = cv::Mat::zeros(1, 128, CV_32FC1);
					cv::Mat d2 = cv::Mat::zeros(1, 128, CV_32FC1);
					for( int k = 0 ; k < 128; k++ )
					{
						d1.at<float>(0, k) = descr1->points[i].siftDescr[k];
						d2.at<float>(0, k) = descr2->points[j].siftDescr[k];
					}
						//dotProduct += descr1->points[i].siftDescr[k] * descr2->points[j].siftDescr[k];
					match_score = acos( d1.row(0).dot(d2.row(0)));
					
					if (match_score < best_score)
					{
						best_idx = j;
						second_best_score = best_score;
						best_score = match_score;
					}
					else if (match_score < second_best_score)
						second_best_score = match_score;
				}
			}
			//std::cerr<<best_score<<" ";
			if( best_score < sift_ratio*second_best_score && best_score < 1 && (sim_score[best_idx] == 0 || sim_score[best_idx] > best_score) )
			{
				sim_score[best_idx] = best_score;
				pcl::Correspondence corr (i, best_idx, best_score);
				model_scene_corrs->push_back (corr);
			}
		}
	}
	for(int j = model_scene_corrs->size()-1 ; j >= 0 ; j-- )
		if( sim_score[model_scene_corrs->at(j).index_match] < model_scene_corrs->at(j).distance )
			model_scene_corrs->erase(model_scene_corrs->begin()+j);

	delete []sim_score;
}

void Model_Builder::Match3DSIFT(pcl::PointCloud<SIFTDescr>::Ptr model_descr, pcl::PointCloud<SIFTDescr>::Ptr scene_descr, pcl::CorrespondencesPtr model_scene_corrs)
{
	bool *model_flags=new bool[model_descr->points.size()];
	bool *scene_flags=new bool[scene_descr->points.size()];

	memset(model_flags, 0, model_descr->points.size()*sizeof(bool));
	memset(scene_flags, 0, scene_descr->points.size()*sizeof(bool));

	Find_SIFT_Corrs(model_descr, scene_descr, model_scene_corrs, model_flags, scene_flags);

	for( int k = 0 ; k < model_scene_corrs->size(); k++ )
	{	
		model_flags[model_scene_corrs->at(k).index_query] = true;
		scene_flags[model_scene_corrs->at(k).index_match] = true;
	}

	pcl::CorrespondencesPtr temp (new pcl::Correspondences ());
	Find_SIFT_Corrs(scene_descr, model_descr, temp, scene_flags, model_flags);

	for(int i = 0 ; i < temp->size(); i++ )
	{
		pcl::Correspondence corr(temp->at(i).index_match, temp->at(i).index_query, temp->at(i).distance);
		model_scene_corrs->push_back(corr);
	}
	//std::cerr<<"Total SIFT Matches: "<<model_scene_corrs->size()<<std::endl;

	delete []model_flags;
	delete []scene_flags;
}

std::vector<int> Model_Builder::GetRawSIFT(pcl::PointCloud<PointXYZRGBIM>::Ptr model, pcl::PointCloud<NormalT>::Ptr model_normals, std::vector<cv::KeyPoint> &model_sift_keypoints, cv::Mat model_2DTo3D, cv::Mat pre_sift_descriptors, 
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sift_cloud, cv::Mat &cur_sift_descriptors)
{
	int num = model_sift_keypoints.size(), pty, ptx, model_idx;
	std::vector<int> indices;
	for( int i = 0 ; i < num ; i++ ){
		cv::Point2f pt = model_sift_keypoints.at(i).pt;
		pty = floor(pt.y);// + 0.5);
		ptx = floor(pt.x);// + 0.5);
				
		model_idx = model_2DTo3D.at<int>(pty, ptx);
		if( model_idx >= 0 )
		{
			indices.push_back(i);
			pcl::PointXYZRGBNormal temp;
			PointXYZRGBIM temp_ori(model->points[model_idx]);
			NormalT temp_normal(model_normals->points[model_idx]);
			temp.x = temp_ori.x;
			temp.y = temp_ori.y;
			temp.z = temp_ori.z;
			temp.rgb = temp_ori.rgb;
			temp.r = temp.r;
			temp.g = temp.g;
			temp.b = temp.b;
			temp.curvature = temp_normal.curvature;
			temp.normal_x = temp_normal.normal_x;
			temp.normal_y = temp_normal.normal_y;
			temp.normal_z = temp_normal.normal_z;
			sift_cloud->points.push_back(temp);
		}
	}
	cv::Mat newDescr = cv::Mat::zeros(indices.size(), 128, CV_32FC1);
	
	for(int i = 0 ; i < indices.size(); i++ )
		pre_sift_descriptors.row(indices[i]).copyTo(newDescr.row(i));

	cur_sift_descriptors = newDescr;

	return indices;
}

bool Model_Builder::TestKeyFrame(int test_id, int last_key_id, Eigen::Matrix4f &initial_guess)
{
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr test_keypoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::copyPointCloud(*model_keypoints_vec.at(test_id), *test_keypoints);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr last_keypoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::copyPointCloud(*model_keypoints_vec.at(last_key_id), *last_keypoints); 

	pcl::PointCloud<SIFTDescr>::Ptr test_descr = model_sift_vec.at(test_id); 
	pcl::PointCloud<SIFTDescr>::Ptr last_key_descr = model_sift_vec.at(last_key_id);

	pcl::CorrespondencesPtr test_key_corrs (new pcl::Correspondences ());
	Match3DSIFT(test_descr, last_key_descr, test_key_corrs);

	/***********************************************CHOST Corrs Acquisition******************************************************/
	//std::cerr<<"Corrs before CSHOT: "<<test_key_corrs->size()<<std::endl;

	pcl::CorrespondencesPtr test_cshot_corrs (new pcl::Correspondences ());
	GetCSHOTCorrs(model_cshot_vec.at(test_id), model_cshot_vec.at(last_key_id), test_cshot_corrs);
	
	float *sim_score = new float[test_keypoints->points.size()];
	int *sim_idx = new int[test_keypoints->points.size()];

	memset(sim_score, 0, sizeof(float)*test_keypoints->points.size());
	for(int i = 0 ; i < test_key_corrs->size() ; i++ )
	{
		int idx = test_key_corrs->at(i).index_query;
		if( sim_score[idx] == 0 || sim_score[idx] > test_key_corrs->at(i).distance)
		{
			sim_score[idx] = test_key_corrs->at(i).distance;
			sim_idx[idx] = i;
		}
	}
	//std::vector<int> update;
	for(int i = 0 ; i < test_cshot_corrs->size() ; i++ )
	{
		int idx = test_cshot_corrs->at(i).index_query;
		if( sim_score[idx] == 0 || (sim_score[idx] > 0.5 && test_cshot_corrs->at(i).distance <= 0.25))
			test_key_corrs->push_back(test_cshot_corrs->at(i));
	}
	std::cerr<<"Corrs after CSHOT: "<<test_key_corrs->size()<<std::endl;
	
	delete []sim_idx;
	delete []sim_score;
	if( false )
	{
		int show_flag = test_key_corrs->size();

		pcl::visualization::PCLVisualizer::Ptr viewer2 (new pcl::visualization::PCLVisualizer("Aligned Model"));
		viewer2->initCameraParameters();
		pcl::PointCloud<PointT>::Ptr model_rgb (new pcl::PointCloud<PointT>());
		pcl::PointCloud<PointT>::Ptr scene_rgb (new pcl::PointCloud<PointT>());
		pcl::copyPointCloud (*model_vec.at(test_id), *model_rgb);
		pcl::copyPointCloud (*model_vec.at(last_key_id), *scene_rgb);

		pcl::PointCloud<PointT>::Ptr model_keypoints_rgb (new pcl::PointCloud<PointT>());
		pcl::PointCloud<PointT>::Ptr scene_keypoints_rgb (new pcl::PointCloud<PointT>());
		pcl::copyPointCloud (*test_keypoints, *model_keypoints_rgb);
		pcl::copyPointCloud (*last_keypoints, *scene_keypoints_rgb);

		viewer2->addPointCloud( scene_rgb, "scene");
		pcl::PointCloud<PointT>::Ptr off_model_rgb (new pcl::PointCloud<PointT>());
		pcl::PointCloud<PointT>::Ptr off_model_keypoints_rgb (new pcl::PointCloud<PointT>());
		pcl::transformPointCloud (*model_rgb, *off_model_rgb, Eigen::Vector3f (-0.5,0,0), Eigen::Quaternionf (1, 0, 0, 0));
		pcl::transformPointCloud (*model_keypoints_rgb, *off_model_keypoints_rgb, Eigen::Vector3f (-0.5,0,0), Eigen::Quaternionf (1, 0, 0, 0));
		viewer2->addPointCloud( off_model_rgb, "model");

		for(int i = 0 ; i < test_key_corrs->size() ; i++ )
		{
			std::ostringstream convert;			// stream used for the conversion
			convert << i;

			PointT temp1 = off_model_keypoints_rgb->points[test_key_corrs->at(i).index_query];
			PointT temp2 = scene_keypoints_rgb->points[test_key_corrs->at(i).index_match];

			if( i < show_flag )
				;//viewer2->addLine(temp1, temp2, 0, 0, 255, "Line "+ convert.str());
			else
			{
				viewer2->addLine(temp1, temp2, 0, 255, 0, "Line "+ convert.str());
				viewer2->spin();
				viewer2->removeAllShapes();
			}
		}
		
	}
	/*pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr test_cshot_keypoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr last_cshot_keypoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	GetCSHOTCorrs(model_vec.at(test_id), model_vec.at(last_key_id), test_cshot_keypoints, last_cshot_keypoints, test_cshot_corrs, model_resolution_vec.at(test_id));
	
	std::cerr<<"Corrs before CSHOT: "<<test_key_corrs->size()<<std::endl;
	for(int i = 0 ; i < test_cshot_corrs->size() ; i++ )
	{
		pcl::Correspondence temp(test_keypoints->points.size(), last_keypoints->points.size(), test_cshot_corrs->at(i).distance);
		test_key_corrs->push_back(temp);

		test_keypoints->points.push_back(test_cshot_keypoints->points[ test_cshot_corrs->at(i).index_query ] );
		last_keypoints->points.push_back(last_cshot_keypoints->points[ test_cshot_corrs->at(i).index_match ] );
	}
	*/

	//std::cerr<<"Corrs After CSHOT: "<<test_key_corrs->size()<<std::endl;
	/****************************************************************************************************************************/

	pcl::Correspondences outputcorrespondences;
	pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal> crsc;
	crsc.setInputSource( test_keypoints );
	crsc.setInputTarget( last_keypoints );
	crsc.setInlierThreshold( ransac_inlier_threshold );						//0.007
	crsc.setMaximumIterations( 2500 );
  
	crsc.setInputCorrespondences( test_key_corrs ); 
	crsc.getCorrespondences( outputcorrespondences );
	initial_guess = crsc.getBestTransformation();
	std::cerr<<"inlier number: "<<outputcorrespondences.size()<<std::endl;

	if( outputcorrespondences.size() < keyframe_inliers )
		return true;
	else
		return false;
}

void Model_Builder::updateFullModel(pcl::PointCloud<PointObj>::Ptr curmodel, pcl::PointCloud<PointObj>::Ptr curSIFT, pcl::PointCloud<SIFTDescr>::Ptr curSIFTDescr, pcl::PointCloud<Descriptor3DType>::Ptr curCSHOTDescr)
{
	AddCloud(fullmodel, curmodel);
	//AddCloud(fullmodel_SIFT, curSIFT);
	int sift_num = curSIFT->points.size();
	for( int i = 0 ; i < sift_num ; i++ )
	{
		fullmodel_SIFT->points.push_back(curSIFT->points[i]);
		SIFTDescr_pool->points.push_back(curSIFTDescr->points[i]);
		CSHOTDescr_pool->points.push_back(curCSHOTDescr->points[i]);
	}

	if( sift_flag.size() == 0 )
		sift_flag.push_back(sift_num - 1);
	else
		sift_flag.push_back(sift_num + sift_flag.at(sift_flag.size()-1));
}

void Model_Builder::AddCloud(pcl::PointCloud<PointObj>::Ptr original, pcl::PointCloud<PointObj>::Ptr cur)
{
	int cur_num = cur->points.size();
	for(int i = 0 ; i < cur_num ; i++ )
		original->points.push_back(cur->points[i]);
}

void Model_Builder::AddSIFTDescr(cv::Mat &ori_pool, const cv::Mat &newDescr)
{
	cv::Mat new_pool = cv::Mat::zeros(ori_pool.rows+newDescr.rows, 128, CV_32FC1);
	for( int i = 0; i < ori_pool.rows ; i++ )
		ori_pool.row(i).copyTo(new_pool.row(i));
	for( int i = 0; i < newDescr.rows ; i++ )
		newDescr.row(i).copyTo(new_pool.row(i+ori_pool.rows));
	ori_pool = new_pool;
}

void Model_Builder::ExtractHue(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_hue)
{
	pcl::copyPointCloud<PointT>(*cloud, *cloud_hue);
	for(int j = 0 ; j < cloud->points.size() ; j++ )
	{		
		int rgb[3] = { cloud->points[j].r, cloud->points[j].g, cloud->points[j].b };
		float hsi[3];
		RGBToHSI(rgb, hsi);
		cloud_hue->points[j].h = hsi[0];
		cloud_hue->points[j].s = hsi[1];
		cloud_hue->points[j].v = hsi[2];
		cloud_hue->points[j].x = cloud->points[j].x;
		cloud_hue->points[j].y = cloud->points[j].y;
		cloud_hue->points[j].z = cloud->points[j].z;
		/*float temp = hsi[0] * 6;
		int index;
		if( temp - floor( temp ) <= 0.5 )
			index = floor(temp);
		else
			index = (int)(ceil(temp)) % 6;
		
		cloud_hue->points[j].r = index; //store the hue class at the r channel
		cloud_hue->points[j].g = 0;
		cloud_hue->points[j].b = 0;
		*/
	}
}

void Model_Builder::RGBToHSI(int rgb[], float hsi[])
{
	double r = rgb[0], g = rgb[1], b = rgb[2];
	
	double num = 0.5 * (r - g + r - b);
	double den = sqrt((r - g)*(r - g) + (r - b)* (g - b));
	double theta = acos(num/(den + EPS));

	if( b > g )
		hsi[0] = 2 * PI - theta;
	else
		hsi[0] = theta;
	hsi[0] = hsi[0] / (2*PI);
	
	if( r + g + b == 0 )
		hsi[1] = 1 - 3*std::min(std::min(r, g),b)/(r+g+b+EPS);
	else
		hsi[1] = 1 - 3*std::min(std::min(r, g),b)/(r+g+b);
	if( hsi[1] == 0 )
		hsi[0] = 0;
	hsi[2] = (r + g + b)/3/255.0;
}

double Model_Builder::computeCloudResolution (const pcl::PointCloud<PointT>::ConstPtr &cloud)
{
	double res = 0.0;
	int n_points = 0;
	int nres;
	std::vector<int> indices (2);
	std::vector<float> sqr_distances (2);
	pcl::search::KdTree<PointT> tree;
	tree.setInputCloud (cloud);

	for (size_t i = 0; i < cloud->size (); ++i)
	{
		if (! pcl_isfinite ((*cloud)[i].x))
			continue;
		//Considering the second neighbor since the first is the point itself.
		nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
		if (nres == 2)
		{
			res += sqrt (sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
		res /= n_points;
	return res;
}

void Model_Builder::ComputeCentroid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
{
	pcl::PointXYZ centroid;
	double cx=0, cy=0, cz=0;
	int num = cloud->points.size();
	for(int i=0; i < num ; i++ )
	{
		cx += cloud->points[i].x;
		cy += cloud->points[i].y;
		cz += cloud->points[i].z;
	}
	centroid.x = cx / num;
	centroid.y = cy / num;
	centroid.z = cz / num;

	center_cloud->points.push_back(centroid);
}

void Model_Builder::ComputeCloudRF(pcl::PointCloud<PointT>::Ptr keypoints, const pcl::PointCloud<NormalT>::Ptr surface_normals, 
	const pcl::PointCloud<PointT>::Ptr surface, pcl::PointCloud<pcl::ReferenceFrame>::Ptr keypoints_rf, float rf_rad)
{
	pcl::BOARDLocalReferenceFrameEstimation<PointT, NormalT, pcl::ReferenceFrame> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch(rf_rad);

	rf_est.setInputCloud (keypoints);
    rf_est.setInputNormals (surface_normals);
    rf_est.setSearchSurface (surface);
    rf_est.compute (*keypoints_rf);
}

void Model_Builder::ComputeShift(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<pcl::ReferenceFrame>::Ptr rf, pcl::PointCloud<pcl::PointXYZ>::Ptr shift, pcl::PointCloud<pcl::PointXYZ>::Ptr centroid)
{
	pcl::PointXYZ center;
	center = centroid->at(0);
	
	int num = keypoints->points.size();
	float ori_x, ori_y, ori_z, dst_x, dst_y, dst_z;

	for( int i =0 ; i < num; i++)
	{
		pcl::PointXYZ temp;
		ori_x = center.x - keypoints->points.at(i).x;
		ori_y = center.y - keypoints->points.at(i).y;
		ori_z = center.z - keypoints->points.at(i).z;
		
		temp.x = ori_x * rf->points.at(i).x_axis[0] + ori_y * rf->points.at(i).x_axis[1] + ori_z * rf->points.at(i).x_axis[2];  
		temp.y = ori_x * rf->points.at(i).y_axis[0] + ori_y * rf->points.at(i).y_axis[1] + ori_z * rf->points.at(i).y_axis[2];
		temp.z = ori_x * rf->points.at(i).z_axis[0] + ori_y * rf->points.at(i).z_axis[1] + ori_z * rf->points.at(i).z_axis[2];

		shift->points.push_back(temp);
	}
}

void Model_Builder::computeNormals(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, float normal_ss)
{
	pcl::NormalEstimationOMP<PointT, NormalT> normal_estimation;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	normal_estimation.setSearchMethod (tree);
	normal_estimation.setNumberOfThreads(8);
	normal_estimation.setRadiusSearch(normal_ss);
	normal_estimation.setInputCloud (cloud);
	normal_estimation.compute (*cloud_normals);
}

void Model_Builder::computeKeyNormals(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<NormalT>::Ptr keypoints_normals, pcl::PointCloud<PointT>::Ptr surface, float normal_ss)
{
	pcl::NormalEstimationOMP<PointT, NormalT> normal_estimation;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	normal_estimation.setSearchMethod (tree);
	normal_estimation.setNumberOfThreads(8);
	normal_estimation.setRadiusSearch(normal_ss);
	normal_estimation.setInputCloud (keypoints);
	normal_estimation.setSearchSurface (surface);
	normal_estimation.compute (*keypoints_normals);
}

std::vector<int> Model_Builder::CropCloud(pcl::PointCloud<PointObj>::Ptr cloud, pcl::PointCloud<PointObj>::Ptr subcloud, float min, float max)
{
	std::vector<int> inliers_idx;
	if( min < 0 || max > 5 )
	{
		pcl::copyPointCloud(*cloud, *subcloud);
		std::cerr<<"Cropper failed due to invalid min OR max"<<std::endl;
		return inliers_idx;
	}
	int num = cloud->points.size ();

	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	coefficients->values.push_back(planeCoef[0]);
    coefficients->values.push_back(planeCoef[1]);
    coefficients->values.push_back(planeCoef[2]);
    coefficients->values.push_back(planeCoef[3]);

	pcl::PointCloud<PointT>::Ptr cloud_rgb (new pcl::PointCloud<PointT>());
	pcl::copyPointCloud(*cloud, *cloud_rgb);
	pcl::PointCloud<PointT>::Ptr cloud_projected (new pcl::PointCloud<PointT>()); 
	pcl::ProjectInliers<PointT> proj;
    proj.setModelType (pcl::SACMODEL_PLANE);
    proj.setInputCloud (cloud_rgb);
	proj.setModelCoefficients (coefficients);
    proj.filter (*cloud_projected);

	for(int i = 0; i < num; ++i)
    {
        //distance from the point to the plane
        float dist = sqrt( (cloud_projected->points[i].x-cloud_rgb->points[i].x) * (cloud_projected->points[i].x-cloud_rgb->points[i].x) +
                           (cloud_projected->points[i].y-cloud_rgb->points[i].y) * (cloud_projected->points[i].y-cloud_rgb->points[i].y) +
                           (cloud_projected->points[i].z-cloud_rgb->points[i].z) * (cloud_projected->points[i].z-cloud_rgb->points[i].z) );

        if ( dist >= min && dist <= max )
		{
            subcloud->points.push_back(cloud->points[i]);
			inliers_idx.push_back(i);
		}
    }
    subcloud->width = subcloud->points.size();
    subcloud->height = 1;

	return inliers_idx;
}


bool Model_Builder::myICP(pcl::PointCloud<PointT>::Ptr model, pcl::PointCloud<PointT>::Ptr scene, pcl::PointCloud<NormalT>::Ptr model_normals, pcl::PointCloud<NormalT>::Ptr scene_normals, 
		Eigen::Matrix4f& initial_guess, Eigen::Matrix4f& rotatranslation, float model_resolution, float scene_resolution)
{
	//if the ICP fail to converge, return false. Otherwise, return true!
	int K = 5, Iter_num = 5000;
	pcl::PointCloud<pcl::PointXYZHSV>::Ptr model_hue(new pcl::PointCloud<pcl::PointXYZHSV>);
	pcl::PointCloud<pcl::PointXYZHSV>::Ptr scene_hue(new pcl::PointCloud<pcl::PointXYZHSV>);
	pcl::PointCloud<PointT>::Ptr buffer1(new pcl::PointCloud<PointT>);
	pcl::PointCloud<pcl::PointXYZHSV>::Ptr buffer2(new pcl::PointCloud<pcl::PointXYZHSV>);
	
	pcl::transformPointCloud(*model, *buffer1, initial_guess);
	ExtractHue(buffer1, model_hue);
	ExtractHue(scene, scene_hue);

	pcl::PointCloud<hsi_hist>::Ptr model_hist(new pcl::PointCloud<hsi_hist>());
	//CloudHSI(model_hue, model_hist, model_resolution);
	pcl::PointCloud<hsi_hist>::Ptr scene_hist(new pcl::PointCloud<hsi_hist>());
	//CloudHSI(scene_hue, scene_hist, scene_resolution);
	
	rotatranslation = initial_guess;
	pcl::search::KdTree<pcl::PointXYZHSV> scene_tree;
	scene_tree.setInputCloud(scene_hue);
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZHSV, pcl::PointXYZHSV> SVD;
	//pcl::registration::TransformationEstimationLM<pcl::PointXYZHSV, pcl::PointXYZHSV> LM;
	double last_error = 100000;
	for( size_t iter = 0 ; iter < Iter_num ; iter++ )
	{
		//In each round of ICP, transform the model_hue cloud
		float diffh, diffs, diffv, diffsum, diffcurvature;
		pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
		for( size_t i = 0 ; i < model_hue->points.size() ; i++ )
		{
			std::vector<int> pointIdx;
			std::vector<float> pointDistance;
		
			if ( scene_tree.radiusSearch (model_hue->points[i], model_resolution*K, pointIdx, pointDistance) > 0 )
			{
				float diffmin = 1000;
				int diffmin_idx = -1;
				for (size_t j = 0; j < pointIdx.size (); j++)
				{
					//std::cerr<<"Finding..."<<scene_hue->points[pointIdx[j]].h <<" "<<model_hue->points[i].h<<std::endl;
					if( pointDistance.at(j) < icp_inlier_threshold )
					{
						diffh = std::min( fabs(scene_hue->points[pointIdx[j]].h  - model_hue->points[i].h), 
							std::min(fabs(scene_hue->points[pointIdx[j]].h - 1 - model_hue->points[i].h), fabs(scene_hue->points[pointIdx[j]].h + 1 - model_hue->points[i].h)));
						diffs = fabs(scene_hue->points[pointIdx[j]].s - model_hue->points[i].s);
						diffv = fabs(scene_hue->points[pointIdx[j]].v - model_hue->points[i].v);
						diffcurvature = fabs( scene_normals->points[pointIdx[j]].curvature - model_normals->points[i].curvature);
						diffsum = diffh*2 + diffs*2 + diffv;
						//diffsum = SimHist(model_hist->points[i].histogram, scene_hist->points[pointIdx[j]].histogram, HNUM+SNUM+INUM);
						//std::cerr<<diffcurvature<<std::endl;
						if( diffcurvature < 0.04 && diffmin > diffsum )
						{
							diffmin = diffsum;
							diffmin_idx = j;
						}
					}	
				}
				//std::cerr<<diffcurvature<<" ";
				//std::cerr<<diffmin<<" ";
				if( diffmin <= 0.1 )
				{
					pcl::Correspondence temp;
					temp.index_query = i;
					temp.index_match = pointIdx[diffmin_idx];
					temp.distance = pointDistance.at(diffmin_idx);
					model_scene_corrs->push_back(temp);
				}
			}
		}
		Eigen::Matrix4f svdRt;
		
		SVD.estimateRigidTransformation(*model_hue, *scene_hue, *model_scene_corrs, svdRt);
		//LM.estimateRigidTransformation(*model_hue, *scene_hue, *model_scene_corrs, svdRt);
		pcl::transformPointCloud(*model_hue, *buffer2, svdRt);
		pcl::copyPointCloud(*buffer2, *model_hue);
		rotatranslation = svdRt * rotatranslation ;
		std::cerr<<"Ratio "<<(model_scene_corrs->size()+0.0) / model_hue->points.size()<<std::endl;
		if( (model_scene_corrs->size()+0.0) / model_hue->points.size() >= 0.2 ) //sufficient inlier found
		{
			size_t corrs = model_scene_corrs->size();
			double this_error=0;
			for( size_t j = 0; j < corrs; j++ )
			{
				pcl::PointXYZHSV model_pt = model_hue->points[model_scene_corrs->at(j).index_query];
				pcl::PointXYZHSV scene_pt = scene_hue->points[model_scene_corrs->at(j).index_match];
				NormalT normal_temp = scene_normals->points[model_scene_corrs->at(j).index_match];
				double diffx = model_pt.x - scene_pt.x, diffy = model_pt.y - scene_pt.y, diffz = model_pt.z - scene_pt.z;
				//double diffx = (model_pt.x - scene_pt.x)*normal_temp.normal_x, 
				//	   diffy = (model_pt.y - scene_pt.y)*normal_temp.normal_y, 
				//	   diffz = (model_pt.z - scene_pt.z)*normal_temp.normal_z;
				double dist = sqrt( diffx*diffx + diffy*diffy + diffz*diffz );

				this_error += dist;
			}
			this_error = this_error / corrs;
			//std::cerr<<"This Error: "<<fabs(this_error - last_error)<<std::endl;
			if( fabs(this_error - last_error) < 0.0000001 )  //Convergence reach
			{
				std::cerr<<"Convergence Reached. Error: "<<this_error<<std::endl;
				std::cerr<<"Iter Num: "<<iter<<std::endl;
				return true;
			}
			else
				last_error = this_error;
		}
	}

	return false;
}

Eigen::Matrix4f Model_Builder::DenseAlign(int model_id, int scene_id, Eigen::Matrix4f &initial_guess, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_region)
{
	pcl::PointCloud<PointXYZRGBIM>::Ptr model = model_raw_vec.at(model_id);
	pcl::PointCloud<PointT>::Ptr model_rgb(new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr model_normals(new pcl::PointCloud<NormalT>());
	pcl::PointCloud<PointT>::Ptr scene_rgb(new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr scene_normals(new pcl::PointCloud<NormalT>());
	
	pcl::copyPointCloud(*model_vec[model_id], *model_rgb);
	pcl::copyPointCloud(*model_vec[model_id], *model_normals);
	pcl::copyPointCloud(*model_vec[scene_id], *scene_rgb);
	pcl::copyPointCloud(*model_vec[scene_id], *scene_normals);
	
	pcl::PointCloud<PointT>::Ptr Filtered_region(new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr Filtered_region_normals(new pcl::PointCloud<NormalT>());
	FilterBoundary(model, Filtered_region);
	float resolution = static_cast<float>(computeCloudResolution(Filtered_region));
	computeNormals(Filtered_region, Filtered_region_normals, resolution * normal_ratio);

	Eigen::Matrix4f rotatranslation = Eigen::Matrix4f::Identity();
	//pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_region(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	if( myICP(Filtered_region, scene_rgb, Filtered_region_normals, scene_normals, initial_guess, rotatranslation, resolution, resolution) == false)
	{
		std::cerr<<"MyICP fails to reach convergence!"<<std::endl;
		return rotatranslation;
	}
	//transform_vec.push_back(rotatranslation);

	pcl::copyPointCloud(*Filtered_region, *model_region);
	pcl::copyPointCloud(*Filtered_region_normals, *model_region);
	return rotatranslation;
}

void Model_Builder::Process(pcl::PointCloud<PointXYZRGBIM>::Ptr raw_model, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_keypoints, 
		pcl::PointCloud<SIFTDescr>::Ptr modelSIFTDescr, pcl::PointCloud<Descriptor3DType>::Ptr modelCSHOTDescr)
{
	model_raw_vec.push_back(raw_model);
	model_vec.push_back(model);
	model_keypoints_vec.push_back(model_keypoints);
	model_sift_vec.push_back(modelSIFTDescr);
	model_cshot_vec.push_back(modelCSHOTDescr);
	
	pcl::PointCloud<PointT>::Ptr model_rgb(new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr model_normals(new pcl::PointCloud<NormalT>());
	pcl::copyPointCloud(*model, *model_rgb);
	pcl::copyPointCloud(*model, *model_normals);

	float model_resolution = static_cast<float>(computeCloudResolution(model_rgb));
	//std::cerr<<"Model Resolution: "<<model_resolution<<std::endl;
	model_resolution_vec.push_back(model_resolution);

	/***************************************************************************************************************/
	/*pcl::SHOTColorEstimationOMP<PointT, NormalT, Descriptor3DType, pcl::ReferenceFrame> descr_est;
	descr_est.setRadiusSearch (model_resolution*CSHOT_ratio);
	descr_est.setNumberOfThreads(8);

	pcl::PointCloud<PointT>::Ptr model_keypoints_rgb(new pcl::PointCloud<PointT>());
	pcl::PointCloud<Descriptor3DType>::Ptr model_cshot_descriptors(new pcl::PointCloud<Descriptor3DType>());

	pcl::copyPointCloud (*model_keypoints, *model_keypoints_rgb);
	descr_est.setInputCloud (model_keypoints_rgb);
	descr_est.setInputNormals (model_normals);
	descr_est.setSearchSurface (model_rgb);
	descr_est.compute (*model_cshot_descriptors);*/
	/***************************************************************************************************************/
	
	if( s_keyframe.size() == 0 )  //No keyframe in the Builder yet
	{
		s_keyframe.push_back(model_vec.size()-1);
		pre_initial_guess = Eigen::Matrix4f::Identity();
		transform_vec.push_back(Eigen::Matrix4f::Identity());
		//std::cerr<<"Keyframe Id: "<<model_vec.size()-1<<std::endl;
			
		//Initialize the full model
		pcl::PointCloud<PointT>::Ptr Filtered_region(new pcl::PointCloud<PointT>);
		pcl::PointCloud<NormalT>::Ptr Filtered_region_normals (new pcl::PointCloud<NormalT>);

		FilterBoundary(raw_model, Filtered_region);
		computeNormals(Filtered_region, Filtered_region_normals, model_resolution*normal_ratio);

		pcl::PointCloud<PointObj>::Ptr temp_full(new pcl::PointCloud<PointObj>());
		pcl::copyPointCloud(*Filtered_region, *temp_full);
		pcl::copyPointCloud(*Filtered_region_normals, *temp_full);

		pcl::PointCloud<PointObj>::Ptr temp_SIFT(new pcl::PointCloud<PointObj>());
		pcl::copyPointCloud(*model_keypoints, *temp_SIFT);

		pcl::PointCloud<pcl::ReferenceFrame>::Ptr model_keypoints_rf(new pcl::PointCloud<pcl::ReferenceFrame> ());
		pcl::PointCloud<PointT>::Ptr model_keypoints_rgb(new pcl::PointCloud<PointT>);
		pcl::copyPointCloud(*model_keypoints, *model_keypoints_rgb); 
		ComputeCloudRF(model_keypoints_rgb, model_normals, model_rgb, model_keypoints_rf, model_resolution*rf_ratio);
		for(int i = 0 ; i < temp_SIFT->points.size() ; i++ )
			memcpy(temp_SIFT->points[i].rf, model_keypoints_rf->points[i].rf, sizeof(float)*9);
		
		updateFullModel(temp_full, temp_SIFT, modelSIFTDescr, modelCSHOTDescr);
		
	}
	else
	{
		int last_key_id = s_keyframe.at(s_keyframe.size()-1);
		int test_id = model_vec.size()-1;
		std::cerr<<"Test Id: "<<test_id<<" "<<last_key_id<<std::endl;

		Eigen::Matrix4f initial_guess;
		if( TestKeyFrame(test_id, last_key_id, initial_guess) == true )
		{
			initial_guess = pre_initial_guess;
			//std::cerr<<initial_guess<<std::endl;
			int cur_key_id = test_id-1;
			TestKeyFrame(test_id, cur_key_id, pre_initial_guess);
			
			//std::cerr<<"Keyframe Id: "<<cur_key_id<<" "<<last_key_id<<std::endl;
			s_keyframe.push_back(cur_key_id);

			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cur_key_region(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
			Eigen::Matrix4f Final_guess = DenseAlign(cur_key_id, last_key_id, initial_guess, cur_key_region);
			transform_vec.push_back(Final_guess);
			
			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cur_keypoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
			pcl::copyPointCloud(*model_keypoints_vec.at(cur_key_id), *cur_keypoints);
			for( int j = transform_vec.size() - 1 ; j >= 0 ; j-- )
			{
				pcl::transformPointCloudWithNormals(*cur_key_region, *cur_key_region, transform_vec.at(j));
				pcl::transformPointCloudWithNormals(*cur_keypoints, *cur_keypoints, transform_vec.at(j));
			}	

			pcl::PointCloud<PointObj>::Ptr temp_full(new pcl::PointCloud<PointObj>());
			pcl::PointCloud<PointObj>::Ptr temp_SIFT(new pcl::PointCloud<PointObj>());
			pcl::copyPointCloud(*cur_key_region, *temp_full);
			pcl::copyPointCloud(*cur_keypoints, *temp_SIFT);

			pcl::PointCloud<pcl::ReferenceFrame>::Ptr cur_keypoints_rf(new pcl::PointCloud<pcl::ReferenceFrame> ());
			pcl::PointCloud<PointT>::Ptr cur_keypoints_rgb(new pcl::PointCloud<PointT>);
			pcl::copyPointCloud(*cur_keypoints, *cur_keypoints_rgb); 
			pcl::PointCloud<PointT>::Ptr cur_rgb(new pcl::PointCloud<PointT>);
			pcl::PointCloud<NormalT>::Ptr cur_normals(new pcl::PointCloud<NormalT>);
			pcl::copyPointCloud(*model_vec[cur_key_id], *cur_rgb); 
			pcl::copyPointCloud(*model_vec[cur_key_id], *cur_normals); 

			ComputeCloudRF(cur_keypoints_rgb, cur_normals, cur_rgb, cur_keypoints_rf, model_resolution_vec[cur_key_id]*rf_ratio);
			for(int i = 0 ; i < temp_SIFT->points.size() ; i++ )
				memcpy(temp_SIFT->points[i].rf, cur_keypoints_rf->points[i].rf, sizeof(float)*9);

			updateFullModel(temp_full, temp_SIFT, model_sift_vec[cur_key_id], model_cshot_vec[cur_key_id]);
		}
		else
			pre_initial_guess = initial_guess;
	}
}

void Model_Builder::AdjustNormals(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, const pcl::PointXYZ &origin)
{
	int num = cloud->points.size();
	float diffx, diffy, diffz, dist, theta;
	for( int i = 0; i < num ; i++ )
	{
		PointT temp = cloud->points[i];
		NormalT temp_normals = cloud_normals->points[i];
		diffx = temp.x - origin.x;
		diffy = temp.y - origin.y;
		diffz = temp.z - origin.z;
		dist = sqrt( diffx*diffx + diffy*diffy + diffz*diffz );
		
		theta = acos( (diffx*temp_normals.normal_x + diffy*temp_normals.normal_y + diffz*temp_normals.normal_z)/dist );
		if( theta > PI/2)
		{
			cloud_normals->points[i].normal_x = -cloud_normals->points[i].normal_x;
			cloud_normals->points[i].normal_y = -cloud_normals->points[i].normal_y;
			cloud_normals->points[i].normal_z = -cloud_normals->points[i].normal_z;
		}
	}
}

void Model_Builder::GenModel()
{
	int i, j;
	std::cerr<<cropHeight<<std::endl;
	std::vector<int> sift_idx;
	CropCloud(fullmodel, truemodel, cropHeight, 1);//0.064
	sift_idx = CropCloud(fullmodel_SIFT, trueSIFT, cropHeight, 1);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//pcl::PointCloud<Descriptor3DType>::Ptr raw_cshot(new pcl::PointCloud<Descriptor3DType>());

	//for( i = 0 ; i < model_cshot_vec.size(); i++ )
	//	for( j = 0 ; j < model_cshot_vec.at(i)->points.size() ; j++)
	//		raw_cshot->points.push_back(model_cshot_vec.at(i)->points[j]);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	sift_num = cv::Mat::zeros(1, sift_flag.size(), CV_32SC1);
	for( i = 0, j = 0; i < sift_idx.size() ; i++ )
	{
		final_sift->points.push_back(SIFTDescr_pool->points[sift_idx[i]]);
		final_cshot->points.push_back(CSHOTDescr_pool->points[sift_idx[i]]);
		//std::cerr<<sift_idx[i]<<std::endl;
		if( sift_idx[i] > sift_flag[j] )
		{
			sift_num.at<int>(0, j) = i-1;
			j++;
		}
	}
	sift_num.at<int>(0,j) = i-1;
	std::cerr<<"Total Feature Points: "<<trueSIFT->points.size()<<std::endl;

	pcl::PointCloud<PointT>::Ptr showmodel(new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr show_normals(new pcl::PointCloud<NormalT>());
	pcl::PointCloud<PointT>::Ptr showSIFT(new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr showSIFT_normals(new pcl::PointCloud<NormalT>());
	pcl::copyPointCloud(*truemodel, *showmodel);
	pcl::copyPointCloud(*trueSIFT, *showSIFT);
	pcl::copyPointCloud(*truemodel, *show_normals);
	pcl::PointCloud<pcl::PointXYZ>::Ptr object_center(new pcl::PointCloud<pcl::PointXYZ>());
	ComputeCentroid(showmodel, object_center);

	//Modify keypoints normals
	computeKeyNormals(showSIFT,showSIFT_normals, showmodel, model_resolution_vec[0]*normal_ratio);
	pcl::copyPointCloud(*showSIFT_normals, *trueSIFT);
	AdjustFinalNormals(trueSIFT, object_center->points[0]);
	pcl::copyPointCloud(*trueSIFT, *showSIFT_normals);
	
	//Copy out keyoints reference frame and compute keypoints shift
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr sift_keypoints_rf(new pcl::PointCloud<pcl::ReferenceFrame> ());
	pcl::PointCloud<pcl::PointXYZ>::Ptr sift_shift( new pcl::PointCloud<pcl::PointXYZ>());
	sift_keypoints_rf->points.resize(trueSIFT->points.size());
	for(int i = 0 ; i < trueSIFT->points.size() ; i++ )
		memcpy(sift_keypoints_rf->points[i].rf, trueSIFT->points[i].rf , sizeof(float)*9);
	ComputeShift(showSIFT, sift_keypoints_rf, sift_shift, object_center);
	for(int i = 0 ; i < trueSIFT->points.size() ; i++ )
		memcpy(trueSIFT->points[i].data_s, sift_shift->points[i].data, sizeof(float)*4);

	pcl::PointCloud<PointT>::Ptr showfullmodel(new pcl::PointCloud<PointT>());
	pcl::copyPointCloud(*fullmodel, *showfullmodel);
	
	viewer->addPointCloud (showfullmodel, "cropped_pivot");
	viewer->addPointCloud (object_center, "center");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255, 255, 0, "center");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "center");

	if( show_object )
	{
		viewer->updatePointCloud (showmodel, "cropped_pivot");
	}

	if( show_keypoints )
	{
		//viewer.addPointCloud (showSIFT, "cropped_SIFT");
		//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 255, "cropped_SIFT");
		//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cropped_SIFT");
		viewer->addPointCloudNormals<PointT, NormalT> (showSIFT, showSIFT_normals, 1, 0.02, "cropped_SIFT");
	}
	if( show_shift )
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr temp_shift( new pcl::PointCloud<pcl::PointXYZ>());
		Voting(showSIFT, sift_keypoints_rf, sift_shift, temp_shift);
		for( size_t i = 0; i < showSIFT->points.size(); i++ )
		{
			std::stringstream ss_line;
			ss_line << "projection_line" << i;

			pcl::PointXYZ temp1, temp2;
			temp1.x = showSIFT->points[i].x + temp_shift->points[i].x;
			temp1.y = showSIFT->points[i].y + temp_shift->points[i].y;
			temp1.z = showSIFT->points[i].z + temp_shift->points[i].z;

			temp2.x = showSIFT->points[i].x; // - scene_shift->points[i].x;
			temp2.y = showSIFT->points[i].y; // - scene_shift->points[i].y;
			temp2.z = showSIFT->points[i].z; // - scene_shift->points[i].z;

			viewer->addLine<pcl::PointXYZ, pcl::PointXYZ> (temp1, temp2, 0, 255, 0, ss_line.str ());
		}
	}

	pcl::PointCloud<PointT>::Ptr reduced_model_rgb (new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr reduced_model_normals (new pcl::PointCloud<NormalT>());
	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(showmodel);
	sor.setLeafSize(model_resolution_vec[0], model_resolution_vec[0], model_resolution_vec[0]);
	sor.filter(*reduced_model_rgb);

	// Create a KD-Tree
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
	pcl::PointCloud<PointT> mls_points;
	pcl::MovingLeastSquares<PointT, PointT> mls;
	// Set parameters
	mls.setInputCloud (reduced_model_rgb);
	mls.setComputeNormals(false);
	mls.setPolynomialFit(true);
	mls.setSearchMethod (tree);
	mls.setSearchRadius (0.01);
	// Reconstruct
	mls.process (mls_points);
	pcl::copyPointCloud(mls_points, *reduced_model_rgb);
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_shift(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr reduced_model_center(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr model_rf(new pcl::PointCloud<pcl::ReferenceFrame> ());
	
	computeNormals(reduced_model_rgb, reduced_model_normals, model_resolution_vec[0]*normal_ratio);
	ComputeCentroid(reduced_model_rgb, reduced_model_center);
	AdjustNormals(reduced_model_rgb, reduced_model_normals, reduced_model_center->points[0]);
	ComputeCloudRF(showSIFT, reduced_model_normals, reduced_model_rgb, model_rf, model_resolution_vec[0]*rf_ratio);
	ComputeShift(showSIFT, model_rf, model_shift, reduced_model_center);
	
	computeKeyNormals(showSIFT, showSIFT_normals, reduced_model_rgb, model_resolution_vec[0]*normal_ratio);
	AdjustNormals(showSIFT, showSIFT_normals, reduced_model_center->points[0]);
	
	pcl::copyPointCloud(*showSIFT, *refined_trueSIFT);
	pcl::copyPointCloud(*showSIFT_normals, *refined_trueSIFT);
	
	for(int k=0 ; k<refined_trueSIFT->points.size() ; k++){
		memcpy(refined_trueSIFT->at(k).rf, model_rf->at(k).rf, sizeof(float)*9);
		memcpy(refined_trueSIFT->at(k).data_s, model_shift->at(k).data, sizeof(float)*4);
	}

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr final_mesh_(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	final_mesh = final_mesh_;

	pcl::copyPointCloud(*reduced_model_rgb, *final_mesh);
	pcl::copyPointCloud(*reduced_model_normals, *final_mesh);

	//pcl::io::savePCDFile(saved_prefix+"_siftkeypoints_f.pcd", *model_keypoints_final, true);
}

void Model_Builder::SaveModel(std::string modelname)
{
	pcl::io::savePCDFile(modelname + "_fullmodel.pcd", *truemodel, true);
	pcl::io::savePCDFile(modelname + "_siftkeypoints.pcd", *trueSIFT, true);
	pcl::io::savePCDFile(modelname + "_siftkeypoints_f.pcd", *refined_trueSIFT, true);
	pcl::io::savePCDFile(modelname + "_siftdescr.pcd", *final_sift, true);
	pcl::io::savePCDFile(modelname + "_cshotdescr.pcd", *final_cshot, true);
	pcl::io::savePCDFile(modelname+"_reduced.pcd", *final_mesh, true);

	cv::FileStorage fsSIFT(modelname+"_aux.xml", cv::FileStorage::WRITE);
	fsSIFT << "SIFTNum" << sift_num;
	fsSIFT << "Resolution" << model_resolution_vec[0];
	fsSIFT.release();
}

void Model_Builder::FilterBoundary(pcl::PointCloud<PointXYZRGBIM>::Ptr cloud, pcl::PointCloud<PointT>::Ptr FilteredCloud)
{
	//int binary_map[480+2*BSTEP][640+2*BSTEP]={0}, row, col;
	float threshold = 0.04;
	int row, col;
	cv::Mat depthmap = cv::Mat::zeros(480+2*BLen, 640+2*BLen, CV_32FC1);
	pcl::PointCloud<PointT>::Ptr cloud_rgb(new pcl::PointCloud<PointT>());
	pcl::copyPointCloud(*cloud, *cloud_rgb);
	
	int num = cloud->points.size();
	for(int i=0 ; i<num; i++ )
		depthmap.at<float>((int)cloud->points[i].imY+BLen, (int)cloud->points[i].imX+BLen) = cloud->points[i].z;

	for(int i=0 ; i<num; i++ ){
		row = (int)cloud->points[i].imY+BLen;
		col = (int)cloud->points[i].imX+BLen;
		
		float zCur = cloud->points[i].z;

		if( fabs(depthmap.at<float>(row-BLen, col)-depthmap.at<float>(row, col)) > threshold || fabs(depthmap.at<float>(row+BLen, col)-depthmap.at<float>(row, col)) > threshold
			|| fabs(depthmap.at<float>(row, col)-depthmap.at<float>(row, col-BLen)) > threshold || fabs(depthmap.at<float>(row, col)-depthmap.at<float>(row, col+BLen)) > threshold)
			;//Boundary->push_back(cloud_rgb->points[i]);
		else
			FilteredCloud->push_back(cloud_rgb->points[i]);
	}
	if( show_region )
	{
		pcl::visualization::PCLVisualizer viewerf ("Filtered Model");
		//viewer.addCoordinateSystem(0.3);
		viewerf.initCameraParameters();
		viewerf.addPointCloud(cloud_rgb, "original");
		pcl::visualization::PointCloudColorHandlerCustom<PointT> region_handler(FilteredCloud, 255, 0, 0);
		viewerf.addPointCloud(FilteredCloud, region_handler, "region");
		viewerf.spin();
	}
	
}

void Model_Builder::AdjustFinalNormals(pcl::PointCloud<PointObj>::Ptr final_cloud, const pcl::PointXYZ &origin)
{
	int num = final_cloud->points.size();
	float diffx, diffy, diffz, dist, theta;
	for( int i = 0; i < num ; i++ )
	{
		PointObj temp = final_cloud->points[i];
		diffx = temp.x - origin.x;
		diffy = temp.y - origin.y;
		diffz = temp.z - origin.z;
		dist = sqrt( diffx*diffx + diffy*diffy + diffz*diffz );
		
		theta = acos( (diffx*temp.normal_x + diffy*temp.normal_y + diffz*temp.normal_z)/dist );
		if( theta > PI/2)
		{
			final_cloud->points[i].normal_x = -final_cloud->points[i].normal_x;
			final_cloud->points[i].normal_y = -final_cloud->points[i].normal_y;
			final_cloud->points[i].normal_z = -final_cloud->points[i].normal_z;
		}
	}
}

void Model_Builder::GetCSHOTCorrs(pcl::PointCloud<Descriptor3DType>::Ptr model_descriptors, pcl::PointCloud<Descriptor3DType>::Ptr scene_descriptors, pcl::CorrespondencesPtr model_scene_corrs)
{
	//std::cerr<<"Sampling Rate: "<<sampling_rate<<" CSHOT Ratio: "<<CSHOT_ratio<<" CSHOT Threshold: "<<CSHOT_threshold<<std::endl;

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

		int found_neighs = match_search.nearestKSearch (model_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
		if(found_neighs == 1 && neigh_sqr_dists[0] < CSHOT_threshold) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
		{
			pcl::Correspondence corr (static_cast<int> (i), neigh_indices[0], neigh_sqr_dists[0]);
			model_scene_corrs->push_back (corr);
		}
	}
	std::cout << "CSHOT Correspondences found: " << model_scene_corrs->size () << std::endl;
	/***********************************************************************************************************************************************************************************/
}


/*
pcl::PointCloud<int> sampled_indices;
	//pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_keypoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	//pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene_keypoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

	if( model_keypoints->points.size() == 0 )
	{
		pcl::UniformSampling<pcl::PointXYZRGBNormal> uniform_sampling;
		uniform_sampling.setInputCloud (model);
		uniform_sampling.setRadiusSearch (model_resolution*sampling_rate);
		uniform_sampling.compute (sampled_indices);
		pcl::copyPointCloud (*model, sampled_indices.points, *model_keypoints);
		//std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
	}
	if( scene_keypoints->points.size() == 0 )
	{
		pcl::UniformSampling<pcl::PointXYZRGBNormal> uniform_sampling;
		uniform_sampling.setInputCloud (scene);
		uniform_sampling.setRadiusSearch (model_resolution*sampling_rate);
		sampled_indices.clear();
		uniform_sampling.compute(sampled_indices);
		pcl::copyPointCloud (*scene, sampled_indices.points, *scene_keypoints);
		//std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;
	}
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

	if( false )
	{
		pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer("Aligned Model"));
		viewer->initCameraParameters();

		viewer->addPointCloud( scene_rgb, "scene");
		pcl::PointCloud<PointT>::Ptr off_model_rgb (new pcl::PointCloud<PointT>());
		pcl::PointCloud<PointT>::Ptr off_model_keypoints_rgb (new pcl::PointCloud<PointT>());
		pcl::transformPointCloud (*model_rgb, *off_model_rgb, Eigen::Vector3f (-0.5,0,0), Eigen::Quaternionf (1, 0, 0, 0));
		pcl::transformPointCloud (*model_keypoints_rgb, *off_model_keypoints_rgb, Eigen::Vector3f (-0.5,0,0), Eigen::Quaternionf (1, 0, 0, 0));
	
		viewer->addPointCloud( off_model_rgb, "model");
		viewer->addPointCloud( off_model_keypoints_rgb, "model_keypoints");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 255, "model_keypoints");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "model_keypoints");
		viewer->addPointCloud( scene_keypoints_rgb, "scene_keypoints");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 255, "scene_keypoints");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

		for(int i = 0 ; i < model_scene_corrs->size() ; i++ )
		{
			std::ostringstream convert;			// stream used for the conversion
			convert << i;

			PointT temp1 = off_model_keypoints_rgb->points[model_scene_corrs->at(i).index_query];
			PointT temp2 = scene_keypoints_rgb->points[model_scene_corrs->at(i).index_match];

			viewer->addLine(temp1, temp2, 0, 255, 0, "Line "+ convert.str());
		
		}
		viewer->spin();
	}
	*/