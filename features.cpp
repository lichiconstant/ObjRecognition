#include "features.h"
#include <pcl/surface/mls.h>

SurfaceNormalsPtr estimateSurfaceNormals (const PointCloudPtr & input, float radius)
{
  //pcl::NormalEstimation<PointT, NormalT> normal_estimation;
  pcl::NormalEstimationOMP<PointT, NormalT> normal_estimation;
  normal_estimation.setSearchMethod (pcl::search::Search<PointT>::Ptr (new pcl::search::KdTree<PointT>));
  normal_estimation.setRadiusSearch (radius);
  normal_estimation.setInputCloud (input);
  SurfaceNormalsPtr normals (new SurfaceNormals);
  normal_estimation.compute (*normals);

  return (normals);
}

/* Use SIFTKeypoint to detect a set of keypoints
 * Inputs:
 *   points
 *     The input point cloud
 *   normals
 *     The input surface normals
 *   min_scale
 *     The smallest scale in the difference-of-Gaussians (DoG) scale-space
 *   nr_octaves
 *     The number of times the scale doubles in the DoG scale-space
 *   nr_scales_per_octave
 *     The number of scales computed for each doubling
 *   min_contrast
 *     The minimum local contrast that must be present for a keypoint to be detected
 * Return: A pointer to a point cloud of keypoints
 */
PointCloudPtr detectKeypoints (const PointCloudPtr & points, const SurfaceNormalsPtr & normals,
                 float min_scale, int nr_octaves, int nr_scales_per_octave, float min_contrast)
{
  pcl::SIFTKeypoint<PointT, pcl::PointWithScale> sift_detect;
  sift_detect.setSearchMethod (pcl::search::Search<PointT>::Ptr (new pcl::search::KdTree<PointT>));
  sift_detect.setScales (min_scale, nr_octaves, nr_scales_per_octave);
  sift_detect.setMinimumContrast (min_contrast);
  sift_detect.setInputCloud (points);
  pcl::PointCloud<pcl::PointWithScale> keypoints_temp;
  sift_detect.compute (keypoints_temp);
  PointCloudPtr keypoints (new PointCloud);
  pcl::copyPointCloud (keypoints_temp, *keypoints);

  return (keypoints);
}

/* Use FPFHEstimation to compute local feature descriptors around each keypoint
 * Inputs:
 *   points
 *     The input point cloud
 *   normals
 *     The input surface normals
 *   keypoints
 *     A cloud of keypoints specifying the positions at which the descriptors should be computed
 *   feature_radius
 *     The size of the neighborhood from which the local descriptors will be computed 
 * Return: A pointer to a LocalDescriptors (a cloud of LocalDescriptorT points)
 */

FPFHDescriptorsPtr computeFPFHDescriptors (const PointCloudPtr & points, const SurfaceNormalsPtr & normals, 
                        const PointCloudPtr & keypoints, float feature_radius)
{
  pcl::FPFHEstimationOMP<PointT, NormalT, FPFHDescriptorT> fpfh_estimation;
  fpfh_estimation.setSearchMethod (pcl::search::Search<PointT>::Ptr (new pcl::search::KdTree<PointT>));
  fpfh_estimation.setRadiusSearch (feature_radius);
  fpfh_estimation.setSearchSurface (points);  
  fpfh_estimation.setInputNormals (normals);
  fpfh_estimation.setInputCloud (keypoints);
  FPFHDescriptorsPtr local_descriptors (new FPFHDescriptors);
  fpfh_estimation.compute (*local_descriptors);

  return (local_descriptors);
}

SHOTDescriptorsPtr computeSHOTDescriptors (const PointCloudPtr & points, const SurfaceNormalsPtr & normals, 
                        const PointCloudPtr & keypoints, float feature_radius)
{ 
  SHOTDescriptorsPtr local_descriptors (new SHOTDescriptors);
  pcl::SHOTEstimationOMP<PointT, NormalT, SHOTDescriptorT> SHOT_estimation;
  SHOT_estimation.setRadiusSearch (feature_radius);

  SHOT_estimation.setInputCloud (keypoints);
  SHOT_estimation.setInputNormals (normals);
  SHOT_estimation.setSearchSurface (points);
  SHOT_estimation.compute (*local_descriptors);

  return (local_descriptors);
}

/* Use VFHEstimation to compute a single global descriptor for the entire input cloud
 * Inputs:
 *   points
 *     The input point cloud
 *   normals
 *     The input surface normals
 * Return: A pointer to a GlobalDescriptors point cloud (a cloud containing a single GlobalDescriptorT point)
 */
GlobalDescriptorsPtr computeGlobalDescriptor (const PointCloudPtr & points, const SurfaceNormalsPtr & normals)
{
  pcl::VFHEstimation<PointT, NormalT, GlobalDescriptorT> vfh_estimation;
  vfh_estimation.setSearchMethod (pcl::search::Search<PointT>::Ptr (new pcl::search::KdTree<PointT>));
  vfh_estimation.setInputCloud (points);
  vfh_estimation.setInputNormals (normals);
  GlobalDescriptorsPtr global_descriptor (new GlobalDescriptors);
  vfh_estimation.compute (*global_descriptor);

  return (global_descriptor);
}

/* Estimate normals, detect keypoints, and compute local and global descriptors 
 * Return: An ObjectFeatures struct containing all the features
 */
ObjectFeatures computeFeatures (const PointCloudPtr & input)
{
  ObjectFeatures features;
  features.points = input;
  features.normals = estimateSurfaceNormals (input, 0.05);
  features.keypoints = detectKeypoints (input, features.normals, 0.005, 10, 8, 1.5);
  features.SHOT_descriptors = computeSHOTDescriptors (input, features.normals, features.keypoints, 0.1);
  features.global_descriptor = computeGlobalDescriptor (input, features.normals);

  return (features);
}

bool ComputeSIFT(pcl::PointCloud<myPointT>::Ptr mycloud, cv::Mat& cloud_image, cv::Mat& cloud_2DTo3D, 
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

void Find_SIFT_Corrs(std::vector< cv::DMatch >& cv_matches, pcl::CorrespondencesPtr model_scene_corrs, 
                     pcl::PointCloud<PointT>::Ptr cloud1, pcl::PointCloud<PointT>::Ptr cloud2,
                     cv::Mat cloud1_2DTo3D, cv::Mat cloud2_2DTo3D,
                     std::vector<cv::KeyPoint> cloud_sift_keypoints1, std::vector<cv::KeyPoint> cloud_sift_keypoints2,
                     cv::Mat cloud_sift_descriptors1, cv::Mat cloud_sift_descriptors2,
                     pcl::PointCloud<PointT>::Ptr cloud_keypoints1, pcl::PointCloud<PointT>::Ptr cloud_keypoints2, float sift_ratio, float *sim_score, bool *flags1, bool *flags2, std::vector< cv::DMatch >& cv_matches_show)
{
    //1 : model
    //2 : scene
	//float *sim_score = new float[cloud2->points.size()];
	//memset(sim_score, 0 , cloud2->points.size()*sizeof(float));

    int ii, jj, best_idx,count_idx=0;
    double best_score,match_score,second_best_score;
    for ( ii=0; ii<cloud_sift_descriptors1.rows ; ii++ )
    {
		if( flags1[ii] == false)
		{
			best_score = 100000;
			second_best_score = 200000;
			for (jj=0; jj<cloud_sift_descriptors2.rows ; jj++ )
			{
				if( flags2[jj] == false )
				{
					match_score = acos( cloud_sift_descriptors1.row(ii).dot(cloud_sift_descriptors2.row(jj)));
					if (match_score < best_score)
					{
						best_idx = jj;
						second_best_score = best_score;
						best_score = match_score;
					}
					else if (match_score < second_best_score)
						second_best_score = match_score;
				}
			}
        
			if( best_score < sift_ratio*second_best_score ) //&& best_score <= 0.6 )
			{
				//std::stringstream ss_line;
				//ss_line << "correspondence_line" << ii ;

				cv::Point2f model_pt = cloud_sift_keypoints1[ii].pt;
				cv::Point2f scene_pt = cloud_sift_keypoints2[best_idx].pt;

				int modely = floor(model_pt.y);// + 0.5);
				int modelx = floor(model_pt.x);// + 0.5);
				int sceney = floor(scene_pt.y);// + 0.5);
				int scenex = floor(scene_pt.x);// + 0.5);

				int model_idx = cloud1_2DTo3D.at<int>(modely, modelx);
				int scene_idx = cloud2_2DTo3D.at<int>(sceney, scenex);
			
				if( (model_idx >=0 && scene_idx >=0) && ( sim_score[scene_idx] == 0 || sim_score[scene_idx] > best_score ))
				{
					sim_score[scene_idx] = best_score;

					cloud_keypoints1->push_back(cloud1->points.at(model_idx));
					cloud_keypoints2->push_back(cloud2->points.at(scene_idx));
            
					cv::DMatch temp_match(model_idx,scene_idx,best_score);
					cv::DMatch temp_match1(ii,best_idx,best_score);
					//std::cerr<<model_idx<<" "<<scene_idx<<" "<<best_score<<std::endl;
					cv_matches.push_back(temp_match);
					cv_matches_show.push_back(temp_match1);

					pcl::Correspondence corr (count_idx, count_idx, best_score);
					model_scene_corrs->push_back (corr);
					count_idx++;
				}
            
			}
		}
    }
	int match_num = cv_matches.size();
	model_scene_corrs->clear();
	cloud_keypoints1->points.clear();
	cloud_keypoints2->points.clear();
	//std::cerr<<"Matches Number 1: "<<scene_keypoints->points.size()<<" "<<model_keypoints->points.size()<<" "<<cv_matches.size()<<std::endl;
	
	for(int j = match_num-1, count_idx = 0 ; j >= 0 ; j-- )
	{
		if( sim_score[cv_matches.at(j).trainIdx] >= cv_matches.at(j).distance )
		{
			cloud_keypoints1->points.push_back(cloud1->points[cv_matches.at(j).queryIdx]);
			cloud_keypoints2->points.push_back(cloud2->points[cv_matches.at(j).trainIdx]);

			pcl::Correspondence corr_temp (count_idx, count_idx, cv_matches.at(j).distance);
			model_scene_corrs->push_back (corr_temp);
			count_idx++;
		}
		else
			cv_matches_show.erase(cv_matches_show.begin()+j);
	}
	//delete []sim_score;
}

void SIFT_Matches(std::vector< cv::DMatch >& cv_matches, std::vector< cv::DMatch >& cv_matches_show, pcl::CorrespondencesPtr model_scene_corrs, 
                     pcl::PointCloud<PointT>::Ptr model_rgb, pcl::PointCloud<PointT>::Ptr scene_rgb,
                     cv::Mat model_2DTo3D, cv::Mat scene_2DTo3D,
                     std::vector<cv::KeyPoint> model_sift_keypoints, std::vector<cv::KeyPoint> scene_sift_keypoints,
                     cv::Mat model_sift_descriptors, cv::Mat scene_sift_descriptors,
                     pcl::PointCloud<PointT>::Ptr model_keypoints, pcl::PointCloud<PointT>::Ptr scene_keypoints, float sift_ratio)
{
	bool *model_flags = new bool[model_sift_keypoints.size()];
	memset(model_flags, 0, sizeof(bool) * model_sift_keypoints.size());
	int model_num = model_rgb->points.size();
	float *model_score = new float[model_num];
	memset(model_score, 0 , model_num*sizeof(float));

	bool *scene_flags = new bool[scene_sift_keypoints.size()];
	memset(scene_flags, 0 , scene_sift_keypoints.size()*sizeof(bool));
	int scene_num = scene_rgb->points.size();
	float *scene_score = new float[scene_num];
	memset(scene_score, 0 , scene_num*sizeof(float));

	Find_SIFT_Corrs(cv_matches, model_scene_corrs, model_rgb, scene_rgb, model_2DTo3D, scene_2DTo3D, model_sift_keypoints, scene_sift_keypoints,
					model_sift_descriptors, scene_sift_descriptors, model_keypoints, scene_keypoints, sift_ratio, scene_score, model_flags, scene_flags, cv_matches_show);
	for( int k = 0 ; k < cv_matches_show.size(); k++ )
	{	
		model_flags[cv_matches_show.at(k).queryIdx] = false;
		scene_flags[cv_matches_show.at(k).trainIdx] = true;
	}
	
	pcl::CorrespondencesPtr scene_model_corrs (new pcl::Correspondences ());
	std::vector< cv::DMatch > cv_matches_1;
	pcl::PointCloud<PointT>::Ptr model_keypoints_1 (new pcl::PointCloud<PointT> ());
	pcl::PointCloud<PointT>::Ptr scene_keypoints_1 (new pcl::PointCloud<PointT> ());
	std::vector< cv::DMatch > cv_matches_show_1;
	Find_SIFT_Corrs(cv_matches_1, scene_model_corrs, scene_rgb, model_rgb, scene_2DTo3D, model_2DTo3D, scene_sift_keypoints, model_sift_keypoints,
					scene_sift_descriptors, model_sift_descriptors, scene_keypoints_1, model_keypoints_1, sift_ratio, model_score, scene_flags, model_flags, cv_matches_show_1);
	//std::cerr<<"New Features Added: "<<model_keypoints_1->points.size()<<" "<<scene_keypoints_1->points.size()<<" "<<cv_matches_show_1.size()<<std::endl;
	int start_idx = model_keypoints->points.size();
	for( int k = 0 ; k < model_keypoints_1->points.size(); k++, start_idx++ )
	{
		model_keypoints->points.push_back(model_keypoints_1->points[k]);
		scene_keypoints->points.push_back(scene_keypoints_1->points[k]);
		pcl::Correspondence corr_temp (start_idx, start_idx, cv_matches_1.at(k).distance);
		model_scene_corrs->push_back(corr_temp);
		cv::DMatch it;
		it.queryIdx = cv_matches_show_1.at(k).trainIdx;
		it.trainIdx = cv_matches_show_1.at(k).queryIdx;
		it.distance = cv_matches_show_1.at(k).distance;
		cv_matches_show.push_back(it);
	}
	std::cerr<<"Total SIFT Matches: "<<model_keypoints->points.size()<<std::endl;
	
	delete []model_score;
	delete []scene_score;
	delete []model_flags;
	delete []scene_flags;
}

void RGBToHSI(int rgb[], float hsi[])
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


double computeCloudResolution (const pcl::PointCloud<PointT>::ConstPtr &cloud)
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

void ComputeCentroid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
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


bool PreProcess(pcl::PointCloud<PointXYZRGBIM>::Ptr model, pcl::PointCloud<pcl::Normal>::Ptr model_normals, cv::Mat& model_image,
				cv::Mat& model_2DTo3D, std::vector<cv::KeyPoint>& model_sift_keypoints, cv::Mat& model_sift_descriptors, pcl::PointCloud<pcl::PointXYZ>::Ptr centroid, float& model_resolution)
{
	float normal_ss;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_rgb (new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::copyPointCloud(*model, *model_rgb);

	model_resolution = static_cast<float> (computeCloudResolution (model_rgb));
	//std::cout << "Model resolution:             " << model_resolution << std::endl;
	if (model_resolution != 0.0f)
		normal_ss = normal_ratio * model_resolution;

	ComputeCentroid(model_rgb, centroid);
	
	computeNormals(model_rgb, model_normals, normal_ss);
	//Extract SIFT keypoints in model
	if (ComputeSIFT(model, model_image, model_2DTo3D, model_sift_keypoints, model_sift_descriptors) == false)
	{
		std::cerr<<"Failed to Compute SIFT Features"<<std::endl;
		return false;
	}

	return true;
}

void computeLocalFeatures(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<PointT>::Ptr surface, pcl::PointCloud<NormalT>::Ptr surface_normals, pcl::PointCloud<pcl::FPFHSignature33>::Ptr features, float resolution)
{
    pcl::FPFHEstimationOMP<PointT, NormalT, pcl::FPFHSignature33> fpfh;

	//fpfh.setNumberOfThreads(8);
	fpfh.setInputCloud(cloud);
	fpfh.setSearchSurface(surface);
	fpfh.setInputNormals(surface_normals);
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
	fpfh.setSearchMethod (tree);

	fpfh.setRadiusSearch(resolution*fpfh_ratio);
	// Compute the features
	fpfh.compute(*features);
}

void computeNormals(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, float normal_ss)
{
	pcl::NormalEstimationOMP<PointT, NormalT> normal_estimation;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	normal_estimation.setSearchMethod (tree);
	//normal_estimation.setNumberOfThreads(8);
	normal_estimation.setRadiusSearch(normal_ss);
	normal_estimation.setInputCloud (cloud);
	normal_estimation.compute (*cloud_normals);
}

void computeKeyNormals(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<NormalT>::Ptr keypoints_normals, pcl::PointCloud<PointT>::Ptr surface, float normal_ss)
{
	pcl::NormalEstimationOMP<PointT, NormalT> normal_estimation;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	normal_estimation.setSearchMethod (tree);
	//normal_estimation.setNumberOfThreads(8);
	normal_estimation.setRadiusSearch(normal_ss);
	normal_estimation.setInputCloud (keypoints);
	normal_estimation.setSearchSurface (surface);
	normal_estimation.compute (*keypoints_normals);
}

void RefineCloud(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<PointT>::Ptr refined_cloud, float radius)
{
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
	
	// Output has the PointNormal type in order to store the normals calculated by MLS
	pcl::PointCloud<pcl::PointXYZRGBNormal> mls_points;

	// Init object (second point type is for the normals, even if unused)
	pcl::MovingLeastSquares<PointT, pcl::PointXYZRGBNormal> mls;
 
	mls.setComputeNormals (true);

	// Set parameters
	mls.setInputCloud (cloud);
	mls.setPolynomialFit (true);
	mls.setSearchMethod (tree);
	mls.setSearchRadius (radius);  //*7 for preprecessing

	// Reconstruct
	mls.process (mls_points);
	refined_cloud->points.clear();
	pcl::copyPointCloud<pcl::PointXYZRGBNormal, PointT>(mls_points, *refined_cloud);
	//pcl::copyPointCloud<pcl::PointXYZRGBNormal, NormalT>(mls_points, *refined_normals);
}

void AddCloud(pcl::PointCloud<PointObj>::Ptr original, pcl::PointCloud<PointObj>::Ptr cur)
{
	int cur_num = cur->points.size();
	for(int i = 0 ; i < cur_num ; i++ )
		original->points.push_back(cur->points[i]);
}

void AddNormal(pcl::PointCloud<NormalT>::Ptr original, pcl::PointCloud<NormalT>::Ptr cur)
{
	int cur_num = cur->points.size();
	for(int i = 0 ; i < cur_num ; i++ )
		original->points.push_back(cur->points[i]);
}

void GetHueDominant(pcl::PointCloud<PointT>::Ptr model, pcl::PointCloud<PointT>::Ptr model_icp, pcl::PointCloud<PointT>::Ptr scene, pcl::PointCloud<PointT>::Ptr scene_icp)
{
	int j, num;
	int max_index = -1, max_count = 0;
	std::vector<std::vector<int>> hue_idx(6); 
	for( j = 0 ; j < scene->points.size() ; j++ )
	{		
		int rgb[3] = { scene->points[j].r, scene->points[j].g, scene->points[j].b };
		float hsi[3];
		RGBToHSI(rgb, hsi);
		float temp = hsi[0] * 6;
		int index;
		if( temp - floor( temp ) <= 0.5 )
			index = floor(temp);
		else
			index = (int)(ceil(temp)) % 6;
		hue_idx.at(index).push_back(j);
		if( hue_idx.at(index).size() > max_count )
		{
			max_index = index;
			max_count = hue_idx.at(index).size();
		}
	}
	//max_index = 0;
	num = hue_idx.at(max_index).size();
	std::cerr<<hue_idx.at(0).size()<<" "<<hue_idx.at(1).size()<<" "<<hue_idx.at(2).size()<<" "<<hue_idx.at(3).size()<<" "<<hue_idx.at(4).size()<<" "<<hue_idx.at(5).size()<<std::endl;
	for( j = 0; j < num ; j++ )
		scene_icp->points.push_back(scene->points[hue_idx.at(max_index).at(j)]);

	for( j = 0 ; j < model->points.size() ; j++ )
	{		
		int rgb[3] = { model->points[j].r, model->points[j].g, model->points[j].b };
		float hsi[3];
		RGBToHSI(rgb, hsi);
		float temp = hsi[0] * 6;
		int index;
		if( temp - floor( temp ) <= 0.5 )
			index = floor(temp);
		else
			index = (int)(ceil(temp)) % 6;
		if( index == max_index )
			model_icp->points.push_back(model->points[j]);
	}
}


void ExtractHue(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_hue)
{
	//
	int num = cloud->points.size();
	pcl::copyPointCloud<PointT>(*cloud, *cloud_hue);
	#pragma omp parallel for firstprivate(cloud_hue)
	for(int j = 0 ; j < num ; j++ )
	{
		pcl::PointXYZHSV &temp = cloud_hue->points[j];
		int rgb[3] = { cloud->points[j].r, cloud->points[j].g, cloud->points[j].b };
		float hsi[3];
		RGBToHSI(rgb, hsi);
		cloud_hue->points[j].h = hsi[0];
		cloud_hue->points[j].s = hsi[1];
		cloud_hue->points[j].v = hsi[2];
	}
}


void CloudHSI(pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_hue, pcl::PointCloud<hsi_hist>::Ptr cloud_hsi, float cloud_resolution)
{
	int ratio = 2;
	float thresh = 0;
	pcl::search::KdTree<pcl::PointXYZHSV> tree;
	tree.setInputCloud(cloud_hue);
	int num = cloud_hue->points.size();
	float hidx, sidx, iidx;
	float h_range = 1.0/HNUM, s_range = 1.0/SNUM, i_range = 1.0/INUM;
	for( int i = 0; i < num ; i++ ){
		std::vector<int> pointIdx;
		std::vector<float> pointDistance;
		hsi_hist buffer;
		cloud_hsi->points.push_back(buffer);
		memset(cloud_hsi->points[i].histogram, 0 , sizeof(float) * (HNUM+SNUM+INUM));

		tree.radiusSearch(cloud_hue->points[i], cloud_resolution * ratio , pointIdx, pointDistance); 
		int count_h=0, count_s=0, count_i = 0;
		for (int j = 0; j < pointIdx.size (); j++){
			pcl::PointXYZHSV pt = cloud_hue->points[pointIdx[j]];
			//Voting hue
			float temph = pt.h + h_range/2;
			if( temph >= 1 )
				temph = temph - 1;
			hidx = temph/h_range;
			cloud_hsi->points[i].histogram[(int)floor(hidx)] = cloud_hsi->points[i].histogram[(int)floor(hidx)] + 1;
			count_h++;
			//Histogram rounding
			if( hidx - floor(hidx) <= thresh )
			{
				if ( floor(hidx) == 0 )
					cloud_hsi->points[i].histogram[HNUM-1] = cloud_hsi->points[i].histogram[HNUM-1] + 1;
				else
					cloud_hsi->points[i].histogram[(int)floor(hidx)-1] = cloud_hsi->points[i].histogram[(int)floor(hidx)-1] + 1;
				count_h++;
			}
			if( ceil(hidx) - hidx <= thresh)
			{
				if ( ceil(hidx) == HNUM )
					cloud_hsi->points[i].histogram[0] = cloud_hsi->points[i].histogram[0] + 1;
				else
					cloud_hsi->points[i].histogram[(int)ceil(hidx)] = cloud_hsi->points[i].histogram[(int)ceil(hidx)] + 1;
				count_h++;
			}
			//Voting Saturation 
			float temps = pt.s;
			if( temps == 1 ) 
				temps = temps - EPS;
			sidx = temps/s_range;
			cloud_hsi->points[i].histogram[HNUM+(int)floor(sidx)] = cloud_hsi->points[i].histogram[HNUM+(int)floor(sidx)] + 1;
			count_s++;
			if( sidx - floor(sidx) <= thresh && floor(sidx) > 0)
			{
				cloud_hsi->points[i].histogram[HNUM+(int)floor(sidx)-1] = cloud_hsi->points[i].histogram[HNUM+(int)floor(sidx)-1] + 1;
				count_s++;
			}
			if( ceil(sidx) - sidx <= thresh && ceil(sidx) < SNUM)
			{
				cloud_hsi->points[i].histogram[HNUM+(int)ceil(sidx)] = cloud_hsi->points[i].histogram[HNUM+(int)ceil(sidx)] + 1;
				count_s++;
			}
			//Voting for Intensity
			float tempi = pt.v;
			if( tempi == 1 ) 
				tempi = tempi - EPS;
			iidx = tempi/i_range;
			cloud_hsi->points[i].histogram[HNUM+SNUM+(int)floor(iidx)] = cloud_hsi->points[i].histogram[HNUM+SNUM+(int)floor(iidx)] + 1;
			count_i++;
			if( iidx - floor(iidx) <= thresh && floor(iidx) > 0)
			{
				cloud_hsi->points[i].histogram[HNUM+SNUM+(int)floor(iidx)-1] = cloud_hsi->points[i].histogram[HNUM+SNUM+(int)floor(iidx)-1] + 1;
				count_i++; 
			}
			if( ceil(iidx) - iidx <= thresh && ceil(iidx) < INUM)
			{
				cloud_hsi->points[i].histogram[HNUM+SNUM+(int)ceil(iidx)] = cloud_hsi->points[i].histogram[HNUM+SNUM+(int)ceil(iidx)] + 1;
				count_i++;
			}
		}
		for( int k = 0 ; k < HNUM ; k++ ){
			cloud_hsi->points[i].histogram[k] = cloud_hsi->points[i].histogram[k]/count_h;
			//std::cerr<<cloud_hsi->points[i].histogram[k]<<" ";
		}
		for( int k = HNUM ; k < HNUM+SNUM ; k++ ){
			cloud_hsi->points[i].histogram[k] = cloud_hsi->points[i].histogram[k]/count_s;
			//std::cerr<<cloud_hsi->points[i].histogram[k]<<" ";
		}
		for( int k = HNUM+SNUM ; k < HNUM+SNUM+INUM ; k++ ){
			cloud_hsi->points[i].histogram[k] = cloud_hsi->points[i].histogram[k]/count_i;
			//std::cerr<<cloud_hsi->points[i].histogram[k]<<" ";
		}
	}
}

float SimHist(float *model_hist, float *scene_hist, int LEN)
{
	float dist=0;
	for( int i = 0 ; i < LEN ; i++ )
		dist = dist + model_hist[i]*scene_hist[i];//dist = dist + sqrt(model_hist[i]*scene_hist[i]);
	return dist;
}

std::vector<int> GetRawSIFT(pcl::PointCloud<PointXYZRGBIM>::Ptr model, pcl::PointCloud<NormalT>::Ptr model_normals, std::vector<cv::KeyPoint> &model_sift_keypoints, cv::Mat model_2DTo3D, cv::Mat pre_sift_descriptors, 
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

void AddSIFTDescr(cv::Mat &ori_pool, const cv::Mat &newDescr)
{
	cv::Mat new_pool = cv::Mat::zeros(ori_pool.rows+newDescr.rows, 128, CV_32FC1);
	for( int i = 0; i < ori_pool.rows ; i++ )
		ori_pool.row(i).copyTo(new_pool.row(i));
	for( int i = 0; i < newDescr.rows ; i++ )
		newDescr.row(i).copyTo(new_pool.row(i+ori_pool.rows));
	ori_pool = new_pool;
}

void AdjustFinalNormals(pcl::PointCloud<PointObj>::Ptr final_cloud, const pcl::PointXYZ &origin)
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

void AdjustNormals(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, const pcl::PointXYZ &origin)
{
	int num = cloud->points.size();
	float diffx, diffy, diffz, dist, theta, norm;
	for( int i = 0; i < num ; i++ )
	{
		PointT temp = cloud->points[i];
		NormalT temp_normals = cloud_normals->points[i];
		diffx = temp.x - origin.x;
		diffy = temp.y - origin.y;
		diffz = temp.z - origin.z;
		dist = sqrt( diffx*diffx + diffy*diffy + diffz*diffz );
		//norm = sqrt( temp_normals.normal_x*temp_normals.normal_x + temp_normals.normal_y*temp_normals.normal_y + temp_normals.normal_z*temp_normals.normal_z );
		theta = acos( (diffx*temp_normals.normal_x + diffy*temp_normals.normal_y + diffz*temp_normals.normal_z)/(dist) );
		if( theta > PI/2)
		{
			cloud_normals->points[i].normal_x = -cloud_normals->points[i].normal_x;
			cloud_normals->points[i].normal_y = -cloud_normals->points[i].normal_y;
			cloud_normals->points[i].normal_z = -cloud_normals->points[i].normal_z;
		}
	}
}

void ComputeCloudRF(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<NormalT>::Ptr surface_normals, 
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

void ComputeShift(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<pcl::ReferenceFrame>::Ptr rf, pcl::PointCloud<pcl::PointXYZ>::Ptr shift, pcl::PointCloud<pcl::PointXYZ>::Ptr centroid)
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

void Voting(pcl::PointCloud<PointT>::Ptr scene_keypoints, pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf, pcl::PointCloud<pcl::PointXYZ>::Ptr model_shift, pcl::PointCloud<pcl::PointXYZ>::Ptr scene_shift)
{
	int num = scene_keypoints->points.size();
	double norm[3];
	for( int i =0 ; i < num; i++)
	{	
		/*
		norm[0] = sqrt(scene_rf->points.at(i).x_axis[0] * scene_rf->points.at(i).x_axis[0]
					 + scene_rf->points.at(i).y_axis[0] * scene_rf->points.at(i).y_axis[0]
					 + scene_rf->points.at(i).z_axis[0] * scene_rf->points.at(i).z_axis[0]);
		norm[1] = sqrt(scene_rf->points.at(i).x_axis[1] * scene_rf->points.at(i).x_axis[1]
					 + scene_rf->points.at(i).y_axis[1] * scene_rf->points.at(i).y_axis[1]
					 + scene_rf->points.at(i).z_axis[1] * scene_rf->points.at(i).z_axis[1]);
		norm[2] = sqrt(scene_rf->points.at(i).x_axis[2] * scene_rf->points.at(i).x_axis[2]
					 + scene_rf->points.at(i).y_axis[2] * scene_rf->points.at(i).y_axis[2]
					 + scene_rf->points.at(i).z_axis[2] * scene_rf->points.at(i).z_axis[2]);
		std::cerr<<norm[0]<<" "<<norm[1]<<" "<<norm[2]<<std::endl;			 
		*/
		pcl::PointXYZ shift = model_shift->points[i];
		pcl::PointXYZ temp;

		temp.x = shift.x * scene_rf->points.at(i).x_axis[0] + shift.y * scene_rf->points.at(i).y_axis[0] + shift.z * scene_rf->points.at(i).z_axis[0];  
		temp.y = shift.x * scene_rf->points.at(i).x_axis[1] + shift.y * scene_rf->points.at(i).y_axis[1] + shift.z * scene_rf->points.at(i).z_axis[1];
		temp.z = shift.x * scene_rf->points.at(i).x_axis[2] + shift.y * scene_rf->points.at(i).y_axis[2] + shift.z * scene_rf->points.at(i).z_axis[2];

		scene_shift->points.push_back(temp);
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


std::vector<int> FilterSIFT(pcl::PointCloud<PointXYZRGBIM>::Ptr model, pcl::PointCloud<NormalT>::Ptr model_normals, std::vector<cv::KeyPoint> &model_sift_keypoints, cv::Mat model_2DTo3D, cv::Mat pre_sift_descriptors, 
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sift_cloud, cv::Mat &cur_sift_descriptors)
{
	int num = model_sift_keypoints.size(), pty, ptx, model_idx;
	std::vector<int> indices;
	std::vector<int> indices2;
	for( int i = 0 ; i < num ; i++ ){
		cv::Point2f pt = model_sift_keypoints.at(i).pt;
		pty = floor(pt.y);// + 0.5);
		ptx = floor(pt.x);// + 0.5);
				
		model_idx = model_2DTo3D.at<int>(pty, ptx);
		if( model_idx >= 0 )
		{
			indices.push_back(model_idx);
			indices2.push_back(i);
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
		pre_sift_descriptors.row(indices2[i]).copyTo(newDescr.row(i));
	
	cur_sift_descriptors = newDescr;

	return indices;
}

void EasySIFT(pcl::PointCloud<myPointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_temp_sift, pcl::PointCloud<SIFTDescr>::Ptr cloud_SIFTDescr, std::vector<int> &sift_idx)
{
	// SIFT Detection
	cv::Mat cloud_image = cv::Mat::zeros(480, 640, CV_8UC1);
	cv::Mat cloud_2DTo3D = cv::Mat::zeros(480,640, CV_32SC1);
	std::vector<cv::KeyPoint> cloud_sift_keypoints;
	cv::Mat old_cloud_descr;
	if (ComputeSIFT(cloud, cloud_image, cloud_2DTo3D, cloud_sift_keypoints, old_cloud_descr) == false)
	{
		std::cerr<<"Failed to Compute SIFT Features"<<std::endl;
		return;
	}
		
	cv::Mat cloud_descr;
		
	sift_idx = FilterSIFT(cloud, cloud_normals, cloud_sift_keypoints, cloud_2DTo3D, old_cloud_descr, cloud_temp_sift, cloud_descr);

	cloud_SIFTDescr->points.resize(cloud_descr.rows);
		
	for( int j = 0 ; j < cloud_descr.rows; j++ )
	{
		for( int k = 0 ; k < cloud_descr.cols; k++ )
			cloud_SIFTDescr->points[j].siftDescr[k] = cloud_descr.at<float>(j, k);
	}
}
