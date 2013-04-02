#include "Recognizer.h"

VotingRecog::VotingRecog(std::string modelname, bool refined_keypoints) : sift_ratio(0.7), resolution(0.00126), bin_size(0.05), normal_ratio(25),
		rf_ratio(25), curvature_threshold(0.03), inlier_threshold(10), PCLViewer(false), show_voting(false), show_surface(false),
		sampling_rate(7), CSHOT_ratio(30), CSHOT_threshold(0.15)
{
	pcl::PointCloud<PointObj>::Ptr model(new pcl::PointCloud<PointObj>);
	std::cerr<<"Loading "<<modelname+"_fullmodel.pcd"<<std::endl;
	pcl::io::loadPCDFile(modelname+"_fullmodel.pcd", *model);
	
	pcl::PointCloud<PointObj>::Ptr model_keypoints_ori(new pcl::PointCloud<PointObj>);
	if( refined_keypoints == true )
	{
		std::cerr<<"Loading "<<modelname+"_siftkeypoints_f.pcd"<<std::endl;
		pcl::io::loadPCDFile(modelname+"_siftkeypoints_f.pcd", *model_keypoints_ori);
	}
	else
	{
		std::cerr<<"Loading "<<modelname+"_siftkeypoints.pcd"<<std::endl;
		pcl::io::loadPCDFile(modelname+"_siftkeypoints.pcd", *model_keypoints_ori);
	}
	pcl::PointCloud<PointT>::Ptr model_keypoints_(new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr model_keypoints_normals_(new pcl::PointCloud<NormalT>());
	pcl::copyPointCloud(*model_keypoints_ori, *model_keypoints_);
	pcl::copyPointCloud(*model_keypoints_ori, *model_keypoints_normals_);
	
	pcl::PointCloud<SIFTDescr>::Ptr model_SIFTDescr_ (new pcl::PointCloud<SIFTDescr> ());
	model_SIFTDescr = model_SIFTDescr_;
	std::cerr<<"Loading "<<modelname+"_siftdescr.pcd"<<std::endl;
	pcl::io::loadPCDFile(modelname+"_siftdescr.pcd", *model_SIFTDescr);

	std::cerr<<"Loading "<<modelname+"_aux.xml"<<std::endl;
	cv::FileStorage fs(modelname+"_aux.xml", cv::FileStorage::READ);
	//fs["SIFTDescr"] >> model_descr;
	fs["SIFTNum"] >> model_sift_num;
	fs["Resolution"] >> resolution;
	fs.release();
	std::cerr<<"Resolution: "<<resolution<<std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr model_shift_(new pcl::PointCloud<pcl::PointXYZ>);
	model_shift_->points.resize(model_keypoints_ori->points.size());
	for(int k=0 ; k<model_keypoints_ori->points.size() ; k++)
		memcpy( model_shift_->at(k).data, model_keypoints_ori->at(k).data_s, sizeof(float)*4);
		
	model_keypoints = model_keypoints_;
	model_keypoints_normals = model_keypoints_normals_;
	model_shift = model_shift_;

	///////////////////////////Filter the original point cloud/////////////////////////
	pcl::PointCloud<PointT>::Ptr model_rgb(new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr model_normals(new pcl::PointCloud<NormalT>());
	pcl::PointCloud<PointT>::Ptr reduced_rgb(new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr reduced_model_normals_(new pcl::PointCloud<NormalT>());
	pcl::copyPointCloud(*model, *model_rgb);
	pcl::copyPointCloud(*model, *model_normals);
	
	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(model_rgb);
	sor.setLeafSize(resolution, resolution, resolution);
	sor.filter(*reduced_rgb);

	reduced_model_rgb = reduced_rgb;
	reduced_model_normals = reduced_model_normals_;

	//pcl::PointCloud<pcl::PointXYZ>::Ptr reduced_model_center(new pcl::PointCloud<pcl::PointXYZ>());
	//computeNormals(reduced_model_rgb, reduced_model_normals, resolution*NORMAL_RATIO);
	//ComputeCentroid(reduced_model_rgb, reduced_model_center);
	//AdjustNormals(reduced_model_rgb, reduced_model_normals, reduced_model_center->points[0]);

	pcl::PointCloud<Descriptor3DType>::Ptr model_CSHOTDescr_ (new pcl::PointCloud<Descriptor3DType> ());
	model_CSHOTDescr = model_CSHOTDescr_;
	std::cerr<<"Loading "<<modelname+"_cshotdescr.pcd"<<std::endl;
	pcl::io::loadPCDFile(modelname+"_cshotdescr.pcd", *model_CSHOTDescr);

	std::cerr<<model_CSHOTDescr->points.size()<<" "<<model_keypoints->points.size()<<" "<<model_SIFTDescr->points.size()<<std::endl;
	//	" "<<model_descr.rows<<std::endl;

	if( refined_keypoints == false )
	{
		model_keypoints_normals->points.clear();
		model_shift->points.clear();

		pcl::PointCloud<pcl::PointXYZ>::Ptr reduced_model_center(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<NormalT>::Ptr reduced_model_normals_(new pcl::PointCloud<NormalT>());
		reduced_model_normals = reduced_model_normals_;

		computeNormals(reduced_model_rgb, reduced_model_normals, resolution*normal_ratio);
		ComputeCentroid(reduced_model_rgb, reduced_model_center);
		AdjustNormals(reduced_model_rgb, reduced_model_normals, reduced_model_center->points[0]);
		pcl::PointCloud<pcl::ReferenceFrame>::Ptr model_rf(new pcl::PointCloud<pcl::ReferenceFrame> ());
		
		ComputeCloudRF(model_keypoints, reduced_model_normals, reduced_model_rgb, model_rf, resolution*rf_ratio);
		ComputeShift(model_keypoints, model_rf, model_shift, reduced_model_center);
	
		computeKeyNormals(model_keypoints, model_keypoints_normals, reduced_model_rgb, resolution*normal_ratio);
		AdjustNormals(model_keypoints, model_keypoints_normals, reduced_model_center->points[0]);
	
		pcl::PointCloud<PointObj>::Ptr model_keypoints_final(new pcl::PointCloud<PointObj>);
		pcl::copyPointCloud(*model_keypoints, *model_keypoints_final);
		pcl::copyPointCloud(*model_keypoints_normals, *model_keypoints_final);
	
		for(int k=0 ; k<model_keypoints_final->points.size() ; k++){
			memcpy(model_keypoints_final->at(k).rf, model_rf->at(k).rf, sizeof(float)*9);
			memcpy(model_keypoints_final->at(k).data_s, model_shift->at(k).data, sizeof(float)*4);
		}
		
		pcl::io::savePCDFile(modelname+"_siftkeypoints_ff.pcd", *model_keypoints_final, true);
	}
}

VotingRecog::~VotingRecog()
{}

void VotingRecog::LoadScene(pcl::PointCloud<PointXYZRGBIM>::Ptr scene, pcl::PointCloud<NormalT>::Ptr scene_normals_)
{
	pcl::PointCloud<PointT>::Ptr scene_rgb_(new pcl::PointCloud<PointT>);
	//pcl::PointCloud<NormalT>::Ptr scene_normals_(new pcl::PointCloud<NormalT>);
	pcl::copyPointCloud(*scene, *scene_rgb_);
	//computeNormals(scene_rgb_, scene_normals_, resolution*normal_ratio);
	
	scene_rgb = scene_rgb_;
	scene_normals = scene_normals_;

	cv::Mat scene_image = cv::Mat::zeros(480, 640, CV_8UC1);
	cv::Mat scene_2DTo3D = cv::Mat::zeros(480,640, CV_32SC1);
	std::vector<cv::KeyPoint> scene_sift_keypoints;
	cv::Mat old_scene_descr;
	if (ComputeSIFT(scene, scene_image, scene_2DTo3D, scene_sift_keypoints, old_scene_descr) == false)
	{
		std::cerr<<"Failed to Compute SIFT Features"<<std::endl;
		return ;
	}
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene_temp_sift(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	cv::Mat scene_descr;
	GetRawSIFT(scene, scene_normals, scene_sift_keypoints, scene_2DTo3D, old_scene_descr, scene_temp_sift, scene_descr);
	pcl::PointCloud<PointT>::Ptr scene_keypoints_(new pcl::PointCloud<PointT>);
	pcl::PointCloud<NormalT>::Ptr scene_keypoints_normals_(new pcl::PointCloud<NormalT>());
	pcl::copyPointCloud(*scene_temp_sift, *scene_keypoints_);
	pcl::copyPointCloud(*scene_temp_sift, *scene_keypoints_normals_);
	scene_keypoints = scene_keypoints_;
	scene_keypoints_normals = scene_keypoints_normals_;

	pcl::PointCloud<SIFTDescr>::Ptr scene_SIFTDescr_(new pcl::PointCloud<SIFTDescr>());
	scene_SIFTDescr = scene_SIFTDescr_;
	scene_SIFTDescr->points.resize(scene_descr.rows);
	for( int j = 0 ; j < scene_descr.rows; j++ )
	{
		for( int k = 0 ; k < scene_descr.cols; k++ )
			scene_SIFTDescr->points[j].siftDescr[k] = scene_descr.at<float>(j, k);
	}

	pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf_(new pcl::PointCloud<pcl::ReferenceFrame> ());
	//pcl::PointCloud<pcl::PointXYZ>::Ptr scene_shift(new pcl::PointCloud<pcl::PointXYZ>);

	ComputeCloudRF(scene_keypoints, scene_normals, scene_rgb, scene_rf_, resolution*rf_ratio);
	scene_rf = scene_rf_;
	std::cerr<<"Scene RF Computation Completed!"<<std::endl;

	pcl::PointCloud<Descriptor3DType>::Ptr scene_CSHOTDescr_(new pcl::PointCloud<Descriptor3DType>());
	scene_CSHOTDescr = scene_CSHOTDescr_;

	pcl::SHOTColorEstimationOMP<PointT, NormalT, Descriptor3DType, pcl::ReferenceFrame> descr_est;
	descr_est.setRadiusSearch (resolution*CSHOT_ratio);
	descr_est.setNumberOfThreads(8);

	descr_est.setInputCloud (scene_keypoints);
	descr_est.setInputNormals (scene_normals);
	descr_est.setSearchSurface (scene_rgb);
	descr_est.compute (*scene_CSHOTDescr);

}

void VotingRecog::LoadScene(pcl::PointCloud<PointT>::Ptr scene_rgb_, pcl::PointCloud<NormalT>::Ptr scene_normals_, 
				pcl::PointCloud<PointT>::Ptr scene_keypoints_, pcl::PointCloud<NormalT>::Ptr scene_keypoints_normals_, 
				pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf_, pcl::PointCloud<SIFTDescr>::Ptr scene_SIFTDescr_, pcl::PointCloud<Descriptor3DType>::Ptr scene_CSHOTDescr_)
{
	scene_rgb = scene_rgb_;
	scene_normals = scene_normals_;
	scene_keypoints = scene_keypoints_;
	scene_keypoints_normals = scene_keypoints_normals_;
	scene_SIFTDescr = scene_SIFTDescr_;
	scene_rf = scene_rf_;

	scene_CSHOTDescr = scene_CSHOTDescr_;
}

void VotingRecog::computeNormals(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, float normal_ss)
{
	pcl::NormalEstimationOMP<PointT, NormalT> normal_estimation;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	normal_estimation.setSearchMethod (tree);
	normal_estimation.setNumberOfThreads(8);
	normal_estimation.setRadiusSearch(normal_ss);
	normal_estimation.setInputCloud (cloud);
	normal_estimation.compute (*cloud_normals);
}

void VotingRecog::computeKeyNormals(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<NormalT>::Ptr keypoints_normals, pcl::PointCloud<PointT>::Ptr surface, float normal_ss)
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

bool VotingRecog::ComputeSIFT(pcl::PointCloud<myPointT>::Ptr mycloud, cv::Mat& cloud_image, cv::Mat& cloud_2DTo3D, 
                 std::vector<cv::KeyPoint>& cloud_sift_keypoints, cv::Mat& cloud_sift_descriptors)
{
    int ii,jj,key_rows,key_cols;
    myPointT tmppoint;
	for( int r = 0 ; r < cloud_2DTo3D.rows ; r++ )
		for( int c = 0 ; c < cloud_2DTo3D.cols ; c++ )
			cloud_2DTo3D.at<int>(r, c) = -1;
    for(ii=0; ii<mycloud->points.size();ii++)
    {
        tmppoint = mycloud->points.at(ii);
        uint32_t rgb = *reinterpret_cast<int*>(&tmppoint.rgb);
                
        uint8_t r = (rgb >> 16) & 0x0000ff;
        uint8_t g = (rgb >> 8)  & 0x0000ff;
        uint8_t b = (rgb)       & 0x0000ff;

        cloud_2DTo3D.at<int>(tmppoint.imY, tmppoint.imX) = ii;
        cloud_image.at<unsigned char>(tmppoint.imY, tmppoint.imX) = 0.2989*(double)r + 0.587*(double)g + 0.114*(double)b;
       
    }
    //Extract Sift Keypoints and descriptors
    cv::imwrite("tmp.pgm",cloud_image);
    system("siftWin32 <tmp.pgm >tmp.key");

    FILE *fp = fopen("tmp.key","r");
    if (!fp)
    {
        std::cerr<<"Cannot open file tmp.key"<<std::endl;
        return false;
    }
    fscanf(fp,"%d %d",&key_rows,&key_cols);
    if(key_cols != 128)
    {
        std::cerr<<"Invalid Keypoint Descriptors"<<std::endl;
        return false;
    }
    cloud_sift_descriptors.create(key_rows,key_cols,CV_32F);

    for ( ii=0; ii<key_rows ; ii++ ) 
    {
        cv::KeyPoint temp_key;
        if (fscanf(fp, "%f %f %f %f", &(temp_key.pt.y), &(temp_key.pt.x), &(temp_key.octave), &(temp_key.angle)) != 4)
        {
            std::cerr<<"Invalid keypoint file format."<<std::endl;
            return false;
        }
        cloud_sift_keypoints.push_back(temp_key);
		//std::cerr<<temp_key.pt.y<<" "<<temp_key.pt.x<<std::endl;
				
        for (jj = 0; jj < key_cols; jj++) 
        {
            int sift_val;
            if (fscanf(fp, "%d", &sift_val) != 1 || sift_val < 0 || sift_val > 255)
	        {
                std::cerr<<"Invalid keypoint value"<<std::endl;
                return false;
            }
	        cloud_sift_descriptors.at<float>(ii,jj) = sift_val;
        }
        cv::normalize(cloud_sift_descriptors.row(ii),cloud_sift_descriptors.row(ii),1);
    }       
    fclose(fp);
    return true;
}


cv::Mat VotingRecog::ExtractRowsCVMat(cv::Mat src, int s_idx, int e_idx)
{
	if( s_idx < 0 || e_idx >= src.rows)
	{
		std::cerr<<"Error Copy Matrix!"<<std::endl;
		cv::Mat newMat;
		return newMat;
	}
	cv::Mat newMat = cv::Mat::zeros(e_idx - s_idx + 1, src.cols, CV_32FC1);
	for( int i = s_idx; i <= e_idx; i++ )
		src.row(i).copyTo(newMat.row(i-s_idx));
	return newMat;
}

void VotingRecog::ComputeCloudRF(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<NormalT>::Ptr surface_normals, 
	pcl::PointCloud<PointT>::Ptr surface, pcl::PointCloud<pcl::ReferenceFrame>::Ptr keypoints_rf, float rf_rad)
{
	pcl::BOARDLocalReferenceFrameEstimation<PointT, NormalT, pcl::ReferenceFrame> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch(rf_rad);

	rf_est.setInputCloud (keypoints);
    rf_est.setInputNormals (surface_normals);
    rf_est.setSearchSurface (surface);
    rf_est.compute (*keypoints_rf);
}

std::vector<int> VotingRecog::GetRawSIFT(pcl::PointCloud<PointXYZRGBIM>::Ptr model, pcl::PointCloud<NormalT>::Ptr model_normals, std::vector<cv::KeyPoint> &model_sift_keypoints, cv::Mat model_2DTo3D, cv::Mat pre_sift_descriptors, 
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

void VotingRecog::FindSurface( pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<PointT>::Ptr scene_rgb, pcl::PointCloud<NormalT>::Ptr scene_normals, pcl::PointCloud<PointT>::Ptr surface)
{
	pcl::KdTreeFLANN<PointT> kdtree;
	kdtree.setInputCloud(scene_rgb);
	//int K = 10;
	float radius = 0.01;
	double theta = 1.3 / 180.0 * M_PI;
	int keypoint_num = keypoints->points.size();
	bool *inliers = new bool[scene_rgb->points.size()];
	memset(inliers, 0, sizeof(bool) * scene_rgb->points.size());

	for( size_t i = 0 ; i < keypoint_num ; i++ )
	{
		std::vector<int> pointIdx1;
		std::vector<float> pointDistance1;
		std::vector<int> list;
		if ( kdtree.nearestKSearch(keypoints->points[i], 1, pointIdx1, pointDistance1) > 0 && pointDistance1.at(0) == 0 && inliers[pointIdx1[0]] == false)
		{
			inliers[pointIdx1[0]] = true;
			list.push_back(pointIdx1[0]);
		}
		else
			continue;

		while(true)
		{
			if(list.size() == 0 )
				break;
			PointT seed = scene_rgb->points[list.at(0)];
			NormalT seed_normal = scene_normals->points[list.at(0)];
			list.erase(list.begin());

			std::vector<int> pointIdx;
			std::vector<float> pointDistance;
			if ( kdtree.radiusSearch (seed, radius, pointIdx, pointDistance) > 0 )
			{
				for (size_t i = 0; i < pointIdx.size (); ++i)
				{
					if (inliers[pointIdx[i]] == false )
					{
						NormalT test_normal = scene_normals->points[pointIdx[i]];
						double temp = seed_normal.normal_x * test_normal.normal_x
									+ seed_normal.normal_y * test_normal.normal_y
									+ seed_normal.normal_z * test_normal.normal_z;
						
						double mag1 = sqrt(seed_normal.normal_x * seed_normal.normal_x
									+ seed_normal.normal_y * seed_normal.normal_y
									+ seed_normal.normal_z * seed_normal.normal_z);

						double mag2 = sqrt(test_normal.normal_x * test_normal.normal_x
									+ test_normal.normal_y * test_normal.normal_y
									+ test_normal.normal_z * test_normal.normal_z);

						if(acos(temp/(mag1*mag2)) < theta)
						{
							inliers[pointIdx[i]] = true;
							list.push_back(pointIdx[i]);
							surface->points.push_back(scene_rgb->points[ pointIdx[i]]);
						}
					}
				}
			}
		}
	}
	delete []inliers;

}

void VotingRecog::ExtractSubDescr(pcl::PointCloud<SIFTDescr>::Ptr ori, pcl::PointCloud<SIFTDescr>::Ptr cur, int s_idx, int e_idx)\
{
	for( int i = s_idx ; i <= e_idx ; i++ )
		cur->points.push_back(ori->points[i]);
}

void VotingRecog::Match3DSIFT(pcl::PointCloud<SIFTDescr>::Ptr descr1, pcl::PointCloud<SIFTDescr>::Ptr descr2, pcl::CorrespondencesPtr scene_model_corrs, float *sim_score)
{
	int num1 = descr1->size();
	int num2 = descr2->size();
	int i, j, best_idx;
	double best_score,match_score,second_best_score;
	for( i = 0 ; i < num1; i++ )
	{
		best_score = 100000;
		second_best_score = best_score+1;
		for( j=0 ; j<num2 ; j++ )
		{
			//match_score = acos( scene_descr.row(i).dot(model_descr.row(j)));
			float dotProduct = 0;
			for( int k = 0 ; k < 128; k++ )
				dotProduct += descr1->points[i].siftDescr[k] * descr2->points[j].siftDescr[k];
			match_score = acos( dotProduct );

			//std::cerr<<match_score<<std::endl;
			if (match_score < best_score)
			{
				best_idx = j;
				second_best_score = best_score;
				best_score = match_score;
			}
			else if (match_score < second_best_score)
				second_best_score = match_score;
		}
		
		if( best_score < sift_ratio*second_best_score && sim_score[i] > best_score )
		{
			pcl::Correspondence corr (i, best_idx, best_score);
			scene_model_corrs->push_back (corr);
			sim_score[i] = best_score;
		}
	}
}

void VotingRecog::ExtractBestCorrs(pcl::CorrespondencesPtr scene_model_corrs)
{
	float *sim_score = new float[scene_SIFTDescr->size()];
	int i,j;
	for(i = 0 ; i < scene_SIFTDescr->size() ; i++ )
		sim_score[i] = 1000;

	std::vector<pcl::CorrespondencesPtr, Eigen::aligned_allocator<pcl::CorrespondencesPtr>> corrs_vec;
	int s_idx=0, e_idx=0;
	for(i = 0 ; i < model_sift_num.cols ; i++ )
	{
		e_idx = model_sift_num.at<int>(0, i);
		if( i != 0 )
			s_idx = model_sift_num.at<int>(0, i-1)+1;

		pcl::PointCloud<SIFTDescr>::Ptr temp_SIFTDescr(new pcl::PointCloud<SIFTDescr>()); 
		ExtractSubDescr(model_SIFTDescr, temp_SIFTDescr, s_idx, e_idx);
		pcl::CorrespondencesPtr corrs_temp( new pcl::Correspondences ());
		Match3DSIFT(scene_SIFTDescr, temp_SIFTDescr, corrs_temp, sim_score);
		for( j = 0 ; j < corrs_temp->size(); j++ )
			corrs_temp->at(j).index_match += s_idx;
		corrs_vec.push_back(corrs_temp);
	}
	for( i = 0; i < corrs_vec.size(); i++ )
	{
		pcl::CorrespondencesPtr corrs_temp = corrs_vec.at(i);
		for( j = 0 ; j < corrs_temp->size(); j++ )
			if( sim_score[corrs_temp->at(j).index_query] >= corrs_temp->at(j).distance )
				scene_model_corrs->push_back(corrs_temp->at(j));
	}
	/*******************************************************************************************************************/
	/*
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr reduced_rgb_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene_rgb_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_key_rgb_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene_key_rgb_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	
	pcl::copyPointCloud(*reduced_model_rgb, *reduced_rgb_normals);
	pcl::copyPointCloud(*reduced_model_normals, *reduced_rgb_normals);
	pcl::copyPointCloud(*scene_rgb, *scene_rgb_normals);
	pcl::copyPointCloud(*scene_normals, *scene_rgb_normals);

	pcl::copyPointCloud(*model_keypoints, *model_key_rgb_normals);
	pcl::copyPointCloud(*model_keypoints_normals, *model_key_rgb_normals);
	pcl::copyPointCloud(*scene_keypoints, *scene_key_rgb_normals);
	pcl::copyPointCloud(*scene_keypoints_normals, *scene_key_rgb_normals);
	*/
	pcl::CorrespondencesPtr cshot_corrs(new pcl::Correspondences());
	GetCSHOTCorrs(scene_CSHOTDescr, model_CSHOTDescr, cshot_corrs);
	
	/*******************************************************************************************************************/
	for(int i = 0 ; i < cshot_corrs->size() ; i++ )
	{
		int idx = cshot_corrs->at(i).index_query;
		if( sim_score[idx] > 0.5 && cshot_corrs->at(i).distance <= 0.25)
			scene_model_corrs->push_back(cshot_corrs->at(i));
	}
	//for( i = 0 ; i < cshot_corrs->size() ; i++ )
	//	scene_model_corrs->push_back(cshot_corrs->at(i));
	
	std::cerr<<"CSHOT+SIFT Correspondences: "<<scene_model_corrs->size()<<std::endl;
	delete []sim_score;
}

void VotingRecog::AdjustNormals(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, const pcl::PointXYZ &origin)
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

std::vector<int> VotingRecog::ClusterOnce(pcl::PointCloud<pcl::PointXYZ>::Ptr votes)
{
	int vote_num = votes->points.size();
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(votes);
	std::vector<int> maxgroup;

	for( int k = 0 ; k < vote_num ; k++ ){
		std::vector<int> pointIdx;
		std::vector<float> pointDistance;
		if ( tree->radiusSearch (votes->points[k], bin_size, pointIdx, pointDistance) > 0 && pointIdx.size() > maxgroup.size())
			maxgroup = pointIdx;
	}
	std::cerr<<"Max Cluster Number: "<<maxgroup.size()<<std::endl;
	return maxgroup;
}

void VotingRecog::VotesClustering(pcl::CorrespondencesPtr corrs_ori, pcl::PointCloud<pcl::PointXYZ>::Ptr votes, std::vector<pcl::CorrespondencesPtr, Eigen::aligned_allocator<pcl::CorrespondencesPtr>> &corrs_clusters)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cur_votes(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::CorrespondencesPtr cur_corrs( new pcl::Correspondences ());
	pcl::copyPointCloud(*votes, *cur_votes);
	*cur_corrs = *corrs_ori;
	
	pcl::ExtractIndices<pcl::PointXYZ> extract1;
	pcl::ExtractIndices<PointT> extract2;
	int i, j;
	while(true)
	{
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
		inliers->indices = ClusterOnce(cur_votes);
		if( inliers->indices.size() < inlier_threshold)
			break;

		pcl::PointCloud<pcl::PointXYZ>::Ptr temp1(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::CorrespondencesPtr temp2( new pcl::Correspondences ());
		pcl::CorrespondencesPtr temp3( new pcl::Correspondences ());

		extract1.setInputCloud (cur_votes);
		extract1.setIndices (inliers);
		extract1.setNegative (true);
		extract1.filter (*temp1);
		cur_votes.swap (temp1);
		
		bool *flag = new bool[cur_corrs->size()];
		memset(flag, 0, sizeof(bool)*cur_corrs->size());
		for( i = 0; i < inliers->indices.size() ; i++ )
			flag[inliers->indices[i]] = true;

		for( i = 0; i < cur_corrs->size() ; i++ )
		{
			if( flag[i] == true )
				temp2->push_back(cur_corrs->at(i));
			else
				temp3->push_back(cur_corrs->at(i));
		}
		delete []flag;

		corrs_clusters.push_back(temp2);
		*cur_corrs = *temp3;
	}
}

void VotingRecog::Reset()
{
	scene_rgb->clear();
	scene_normals->clear();
	scene_keypoints->clear();
	scene_keypoints_normals->clear();
	scene_rf->clear();

	matched_model_keypoints->clear();
	matched_scene_keypoints->clear();

	corrs_clusters.clear();
	candidates.clear();
	transform_vec.clear();
	fitness_score_vec.clear();

	matched_corrs->clear();
}

void VotingRecog::Recognize()
{
	//////////////////////////////////////SIFT Matching/////////////////////////////////////////////////////////////////////////

	pcl::CorrespondencesPtr scene_model_corrs( new pcl::Correspondences ());
	//std::cerr<<"SIFT Matching..."<<std::endl;
	ExtractBestCorrs( scene_model_corrs);
	//std::cerr<<"SIFT Matching Done!"<<std::endl;
		
	pcl::PointCloud<PointT>::Ptr matched_model_keypoints_(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr matched_scene_keypoints_(new pcl::PointCloud<PointT>);
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr matched_scene_rf(new pcl::PointCloud<pcl::ReferenceFrame> ());
	pcl::PointCloud<pcl::PointXYZ>::Ptr matched_model_shift(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr matched_scene_shift(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::CorrespondencesPtr matched_corrs_( new pcl::Correspondences ());

	for(int k=0, count=0 ; k<scene_model_corrs->size() ; k++)
	{
		int scene_idx = scene_model_corrs->at(k).index_query;
		int model_idx = scene_model_corrs->at(k).index_match;
		//std::cerr<<scene_idx<<" "<<model_idx<<std::endl;
		//std::cerr<<fabs(scene_keypoints_normals->points[scene_idx].curvature - model_keypoints_normals->points[model_idx].curvature )<<std::endl;
		if( fabs(scene_keypoints_normals->points[scene_idx].curvature - model_keypoints_normals->points[model_idx].curvature ) < curvature_threshold )
		{
			matched_scene_keypoints_->points.push_back( scene_keypoints->points[scene_idx] );
			matched_scene_rf->points.push_back( scene_rf->points[scene_idx] );
			matched_model_shift->points.push_back( model_shift->points[model_idx] );
			matched_model_keypoints_->points.push_back( model_keypoints->points[model_idx] );
			pcl::Correspondence temp(count, count, scene_model_corrs->at(k).distance);
			matched_corrs_->push_back(temp);
			count++;
		}
	}
	matched_corrs = matched_corrs_;
	matched_model_keypoints = matched_model_keypoints_;
	matched_scene_keypoints = matched_scene_keypoints_;

	std::cerr<<"Final Corrs: "<<matched_model_keypoints->points.size()<<std::endl;
	
	Voting(matched_scene_keypoints, matched_scene_rf, matched_model_shift, matched_scene_shift);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	pcl::PointCloud<PointT>::Ptr off_model_rgb(new pcl::PointCloud<PointT>());
	pcl::PointCloud<PointT>::Ptr off_model_keypoints(new pcl::PointCloud<PointT>());
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr votes_(new pcl::PointCloud<pcl::PointXYZ>());
	for( size_t i = 0; i < matched_scene_keypoints->points.size(); i++ )
	{
		std::stringstream ss_line;
		ss_line << "projection" << i;

		pcl::PointXYZ temp1, temp2;
		temp1.x = matched_scene_keypoints->points[i].x + matched_scene_shift->points[i].x;
		temp1.y = matched_scene_keypoints->points[i].y + matched_scene_shift->points[i].y;
		temp1.z = matched_scene_keypoints->points[i].z + matched_scene_shift->points[i].z;
		votes_->push_back(temp1);

		temp2.x = matched_scene_keypoints->points[i].x; // - scene_shift->points[i].x;
		temp2.y = matched_scene_keypoints->points[i].y; // - scene_shift->points[i].y;
		temp2.z = matched_scene_keypoints->points[i].z; // - scene_shift->points[i].z;
		
		if( PCLViewer && show_voting)
			viewer->addLine<pcl::PointXYZ, pcl::PointXYZ> (temp1, temp2, 255, 255, 0, ss_line.str ());
	}
	votes = votes_;
	
	if( PCLViewer && show_voting)
	{
		viewer->spin();
		viewer->removeAllShapes();
	}

	VotesClustering(matched_corrs, votes, corrs_clusters);
	
	for( size_t k = 0 ; k < corrs_clusters.size() ; k++ ){
		std::stringstream view_label;
		view_label << "projection_line" << k;

		pcl::CorrespondencesPtr temp_corrs( new pcl::Correspondences ());
		temp_corrs = corrs_clusters.at(k);
		
		Eigen::Matrix4f initial_guess, Final_guess;
		pcl::Correspondences outputcorrespondences;
		pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> crsc;
		crsc.setInputSource( matched_scene_keypoints );
		crsc.setInputTarget( matched_model_keypoints );
		crsc.setInlierThreshold( 0.01 );
		crsc.setMaximumIterations( 2000 );
  
		crsc.setInputCorrespondences( temp_corrs ); 
		crsc.getCorrespondences( outputcorrespondences );
		initial_guess = crsc.getBestTransformation();
		std::cerr<<"Transformation Inlier Number: "<<outputcorrespondences.size()<<std::endl;

		if( initial_guess != Eigen::Matrix4f::Identity() && outputcorrespondences.size() >= 4)
		{
			pcl::PointCloud<PointT>::Ptr Final_inliers(new pcl::PointCloud<PointT>());
			for (size_t j = 0; j < outputcorrespondences.size(); j++)
				Final_inliers->push_back(matched_scene_keypoints->at(outputcorrespondences.at(j).index_query));
				
			pcl::PointCloud<PointT>::Ptr Final_surface(new pcl::PointCloud<PointT>());
			FindSurface(Final_inliers, scene_rgb, scene_normals, Final_surface);

			if( Final_surface->points.size() >= 1000 )
			{
				if( PCLViewer && show_surface )
				{
					pcl::visualization::PointCloudColorHandlerCustom<PointT> surface_handler (Final_surface, 255, 128, 0);
					viewer->addPointCloud( Final_surface, surface_handler, "Surface"+view_label.str());
					viewer->addPointCloud( Final_inliers, "Final_inliers"+view_label.str());
					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 255, 0, "Final_inliers"+view_label.str());
					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "Final_inliers"+view_label.str());
					viewer->spin();
				}

				pcl::IterativeClosestPoint<PointT, PointT> icp;
				icp.setMaxCorrespondenceDistance(0.01);
				icp.setTransformationEpsilon (1e-7);
				icp.setMaximumIterations(1000);
				//icp.setInputSource(reduced_model_rgb);
				//icp.setInputTarget(Final_surface);
				icp.setInputSource(Final_surface);
				icp.setInputTarget(reduced_model_rgb);
	
				pcl::PointCloud<PointT>::Ptr Final (new pcl::PointCloud<PointT>);
				icp.align(*Final, initial_guess);
				Final_guess = icp.getFinalTransformation().inverse();

				std::cerr<<"ICP Convergence: "<<icp.hasConverged()<<" Fitness Score: "<<icp.getFitnessScore()<<std::endl;
				if( icp.getFitnessScore() < 0.000005 )
				{
					pcl::PointCloud<PointT>::Ptr candidate(new pcl::PointCloud<PointT>());
					//inv_guess = initial_guess.inverse();
					pcl::transformPointCloud(*reduced_model_rgb, *candidate, Final_guess);
					
					bool isoverlap = false;
					for( size_t cidx = 0 ; cidx < candidates.size() ; cidx ++)
						if( IsOverlap( candidates.at(cidx), candidate)){
							isoverlap = true;
							break;
						}
					if( isoverlap == false )
					{
						candidates.push_back(candidate);
						transform_vec.push_back(Final_guess);
						fitness_score_vec.push_back(icp.getFitnessScore());
					}
				}
			}
		}		
	}
}

bool VotingRecog::IsOverlap( pcl::PointCloud<PointT>::Ptr cloud1, pcl::PointCloud<PointT>::Ptr cloud2)
{
	float xmax=-100, xmin=100, ymax=-100, ymin=100, zmax=-100, zmin=100;
	int num1 = cloud1->points.size();
	int num2 = cloud2->points.size();
	for(int i = 0 ; i < num1 ; i++ ){
		PointT temp = cloud1->points[i];
		if( temp.x > xmax )
			xmax = temp.x;
		if( temp.x < xmin )
			xmin = temp.x;
		if( temp.y > ymax )
			ymax = temp.y;
		if( temp.y < ymin )
			ymin = temp.y;
		if( temp.z > zmax )
			zmax = temp.z;
		if( temp.z < zmin )
			zmin = temp.z;
	}
	for(int i = 0 ; i < num2; i++ ){
		PointT temp = cloud2->points[i];
		if( temp.x <= xmax && temp.x >= xmin &&
			temp.y <= ymax && temp.y >= ymin &&
			temp.z <= zmax && temp.z >= zmin )
			return true;
	}
	return false;
}

void VotingRecog::GetCSHOTCorrs(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene, 
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_keypoints, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene_keypoints, pcl::CorrespondencesPtr model_scene_corrs, float model_resolution)
{
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
	}
	if( scene_keypoints->points.size() == 0 )
	{
		pcl::UniformSampling<pcl::PointXYZRGBNormal> uniform_sampling;
		uniform_sampling.setInputCloud (scene);
		uniform_sampling.setRadiusSearch (model_resolution*sampling_rate);
		sampled_indices.clear();
		uniform_sampling.compute(sampled_indices);
		pcl::copyPointCloud (*scene, sampled_indices.points, *scene_keypoints);
	}
	
	pcl::PointCloud<PointT>::Ptr scene_rgb (new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr scene_normals (new pcl::PointCloud<NormalT>());
	pcl::copyPointCloud (*scene, *scene_rgb);
	pcl::copyPointCloud (*scene, *scene_normals);
	
	pcl::PointCloud<PointT>::Ptr scene_keypoints_rgb (new pcl::PointCloud<PointT>());
	pcl::copyPointCloud (*scene_keypoints, *scene_keypoints_rgb);

	/*******************************************************************Find Correspondences**********************************************************************************************/
	//pcl::PointCloud<Descriptor3DType>::Ptr model_descriptors (new pcl::PointCloud<Descriptor3DType> ());
	pcl::PointCloud<Descriptor3DType>::Ptr scene_descriptors (new pcl::PointCloud<Descriptor3DType> ());

	pcl::SHOTColorEstimationOMP<PointT, NormalT, Descriptor3DType, pcl::ReferenceFrame> descr_est;
	descr_est.setRadiusSearch (model_resolution*CSHOT_ratio);
	descr_est.setNumberOfThreads(8);

	if( model_CSHOTDescr->points.size() == 0 )
	{
		pcl::PointCloud<PointT>::Ptr model_rgb (new pcl::PointCloud<PointT>());
		pcl::PointCloud<NormalT>::Ptr model_normals (new pcl::PointCloud<NormalT>());
		pcl::PointCloud<PointT>::Ptr model_keypoints_rgb (new pcl::PointCloud<PointT>());
		pcl::copyPointCloud (*model, *model_rgb);
		pcl::copyPointCloud (*model_keypoints, *model_keypoints_rgb);
		pcl::copyPointCloud (*model, *model_normals);

		descr_est.setInputCloud (model_keypoints_rgb);
		descr_est.setInputNormals (model_normals);
		descr_est.setSearchSurface (model_rgb);
		descr_est.compute (*model_CSHOTDescr);
	}

	descr_est.setInputCloud (scene_keypoints_rgb);
	descr_est.setInputNormals (scene_normals);
	descr_est.setSearchSurface (scene_rgb);
	descr_est.compute (*scene_descriptors);

	//
	//  Find Model-Scene Correspondences with KdTree
	//
	//pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

	pcl::KdTreeFLANN<Descriptor3DType> match_search;
	match_search.setInputCloud (model_CSHOTDescr);

	//  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
	for (size_t i = 0; i < scene_descriptors->size (); ++i)
	{
		std::vector<int> neigh_indices (1);
		std::vector<float> neigh_sqr_dists (1);
		if (!pcl_isfinite (scene_descriptors->at(i).descriptor[0])) //skipping NaNs
		{
			std::cerr<<"Hi"<<std::endl;
			continue;
		}

		int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
		if(found_neighs == 1 && neigh_sqr_dists[0] < CSHOT_threshold) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
		{
			pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
			model_scene_corrs->push_back (corr);
		}
	}
	std::cout << "CSHOT Correspondences found: " << model_scene_corrs->size () << std::endl;
	/***********************************************************************************************************************************************************************************/
}

void VotingRecog::GetCSHOTCorrs(pcl::PointCloud<Descriptor3DType>::Ptr model_descriptors, pcl::PointCloud<Descriptor3DType>::Ptr scene_descriptors, pcl::CorrespondencesPtr model_scene_corrs)
{
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
		//std::cerr<<neigh_sqr_dists[0]<<std::endl;
		if(found_neighs == 1 && neigh_sqr_dists[0] < CSHOT_threshold) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
		{
			pcl::Correspondence corr (static_cast<int> (i), neigh_indices[0], neigh_sqr_dists[0]);
			model_scene_corrs->push_back (corr);
		}
	}
	std::cout << "CSHOT Correspondences found: " << model_scene_corrs->size () << std::endl;
}
